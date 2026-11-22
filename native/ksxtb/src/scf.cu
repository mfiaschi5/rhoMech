// ksxtb stage 3: the SCF's memory-bound steps on the padded batch, fused.
//
// Matrices are (B, n, n) row-major, one block per molecule, n the batch's largest AO count;
// the multipole integrals (B, 3|6, n, n) as stage 1 centres them: D[c, mu, nu] on atom(nu).
// Each kernel reads S, D and Q once per call instead of torch's ten elementwise passes.
#include <cuda_runtime.h>

#include "ksxtb.h"

namespace {

constexpr int kThreads = 256;

// F = H0 - S (v_mu + v_nu) / 2 - (W + W^T) / 2, W[mu, nu] = D[:, mu, nu] . vdp(atom nu)
//     + Q[:, mu, nu] . vqp(atom nu); padded AOs get `pad` on the diagonal.
// One thread per (molecule of idx, mu, nu).
__global__ void __launch_bounds__(kThreads) fock_kernel(KsFock f) {
  const long long n = f.n, nn = n * n;
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)f.nsel * nn) return;
  const long long b = f.idx[t / nn], e = t % nn, mu = e / n, nu = e % n;
  const long long blk = b * nn;
  double* out = f.fock + t;
  if (!f.valid[b * n + mu] || !f.valid[b * n + nu]) {
    *out = (mu == nu) ? f.pad : 0.0;
    return;
  }
  const double* vm = f.vmp + (b * n + mu) * 9;  // vdp (3) then vqp (6) at atom(mu)
  const double* vn = f.vmp + (b * n + nu) * 9;
  double w = 0.0;
  for (int c = 0; c < 3; ++c) {
    const double* d = f.D + (b * 3 + c) * nn;
    w += d[mu * n + nu] * vn[c] + d[nu * n + mu] * vm[c];
  }
  for (int c = 0; c < 6; ++c) {
    const double* q = f.Q + (b * 6 + c) * nn;
    w += q[mu * n + nu] * vn[3 + c] + q[nu * n + mu] * vm[3 + c];
  }
  const double vao = f.vao[b * n + mu] + f.vao[b * n + nu];
  *out = f.H0[blk + e] - 0.5 * f.S[blk + e] * vao - 0.5 * w;
}

// per AO nu of each molecule: the Mulliken population sum_mu P S and the atomic multipole
// contributions -sum_mu P D[:, mu, nu], -sum_mu P Q[:, mu, nu] (out[(b, nu) * 10 + k])
__global__ void __launch_bounds__(kThreads) mulliken_kernel(KsMulliken m) {
  const long long n = m.n, nn = n * n;
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)m.nmol * n) return;
  const long long b = t / n, nu = t % n;
  double acc[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const double* p = m.P + b * nn + nu;
  const double* s = m.S + b * nn + nu;
  for (long long mu = 0; mu < n; ++mu) {
    double pm = p[mu * n];
    acc[0] += pm * s[mu * n];
    for (int c = 0; c < 3; ++c) acc[1 + c] -= pm * m.D[(b * 3 + c) * nn + mu * n + nu];
    for (int c = 0; c < 6; ++c) acc[4 + c] -= pm * m.Q[(b * 6 + c) * nn + mu * n + nu];
  }
  for (int k = 0; k < 10; ++k) m.out[t * 10 + k] = acc[k];
}

// tblite's get_fermi_filling (aufbau start, Newton steps on the Fermi level until the electron
// count is within sqrt(eps), at most 200) and get_electronic_entropy for one (molecule, spin)
// per warp; emo sorted ascending, nel the channel's electron count.
__global__ void __launch_bounds__(32) fermi_kernel(KsFermi f) {
  const int job = blockIdx.x, lane = threadIdx.x, n = f.n;
  if (job >= f.njob) return;
  const double* e = f.emo + (long long)f.mol[job] * n;
  double* occ = f.occ + (long long)job * n;
  const double nel = f.nel[job], kt = f.kt;
  int homo = (int)floor(nel);
  const double frac = fmod(nel, 1.0);
  for (int i = lane; i < n; i += 32) occ[i] = i < homo ? 1.0 : (i == homo ? frac : 0.0);
  homo = frac > 0.5 ? homo + 1 : homo;
  if (homo > 0) {
    double ef = 0.5 * (e[max(homo, 1) - 1] + e[min(homo + 1, n) - 1]);
    const double thr = 1.4901161193847656e-08, tiny = 1.4916681462400413e-154;
    for (int cycle = 0; cycle < 200; ++cycle) {
      double tot = 0.0, dsum = 0.0;
      for (int i = lane; i < n; i += 32) {
        double x = (e[i] - ef) / kt, fi = 0.0, di = 0.0;
        if (x < 50.0) {
          double ex = exp(x);
          fi = 1.0 / (ex + 1.0);
          di = ex / (kt * (ex + 1.0) * (ex + 1.0));
        }
        occ[i] = fi;
        tot += fi;
        dsum += di;
      }
      for (int o = 16; o > 0; o >>= 1) {
        tot += __shfl_xor_sync(0xffffffffu, tot, o);
        dsum += __shfl_xor_sync(0xffffffffu, dsum, o);
      }
      ef += dsum > tiny ? (homo - tot) / dsum : 0.0;
      if (fabs(homo - tot) <= thr) break;
    }
  }
  __syncwarp();
  double s = 0.0;
  for (int i = lane; i < n; i += 32) {
    double o = occ[i];
    if (o > 0.0 && o < 1.0) s += o * log(o) + (1.0 - o) * log(1.0 - o);
  }
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) f.ts[job] = kt * s;
}

inline unsigned blocks(long long n) { return (unsigned)((n + kThreads - 1) / kThreads); }

}  // namespace

extern "C" int ksxtb_fock(const KsFock* fp, void* stream) {
  const KsFock f = *fp;
  long long total = (long long)f.nsel * f.n * f.n;
  if (total) fock_kernel<<<blocks(total), kThreads, 0, (cudaStream_t)stream>>>(f);
  return (int)cudaGetLastError();
}

extern "C" int ksxtb_mulliken(const KsMulliken* mp, void* stream) {
  const KsMulliken m = *mp;
  long long total = (long long)m.nmol * m.n;
  if (total) mulliken_kernel<<<blocks(total), kThreads, 0, (cudaStream_t)stream>>>(m);
  return (int)cudaGetLastError();
}

extern "C" int ksxtb_fermi(const KsFermi* fp, void* stream) {
  const KsFermi f = *fp;
  if (f.njob) fermi_kernel<<<f.njob, 32, 0, (cudaStream_t)stream>>>(f);
  return (int)cudaGetLastError();
}
