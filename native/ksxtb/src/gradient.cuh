// The H0 gradient contraction (tblite's get_hamiltonian_gradient, closed shell), included by
// integrals.cu. Per off-site AO pair (mu on A, nu on B) the derivatives of S, D and Q with
// respect to A are contracted on the fly with per-pair weights, so no derivative matrix is
// ever stored. The B-centred multipoles are folded in through (r-B) = (r-A) + (A-B):
//   vdp_B . dD_B = vdp_B . dD_A + vdp_B,e S + (vdp_B . v) dS
//   V_B : dQ_B = V_B : dQ_A + 2 (V_B v) . dD_A + 2 (V_B D_A)_e + 2 (V_B v)_e S + (v V_B v) dS
// with V_B the symmetric matrix of the cartesian quadrupole weights of atom B.
#pragma once

namespace {

// Weights of the cartesian second moments (xx xy yy xz yz zz) equivalent to contracting the
// traceless quadrupole with vqp.
__device__ __forceinline__ void cart_quad_weights(const double* vqp, double w[6]) {
  const double tr = 0.5 * (vqp[0] + vqp[2] + vqp[5]);
#pragma unroll
  for (int c = 0; c < 6; ++c) w[c] = 1.5 * vqp[c];
  w[0] -= tr;
  w[2] -= tr;
  w[5] -= tr;
}

// (V x) for the symmetric matrix of the 6 cartesian weights (off-diagonals halved).
__device__ __forceinline__ void quad_mv(const double w[6], const double x[3], double y[3]) {
  y[0] = w[0] * x[0] + 0.5 * w[1] * x[1] + 0.5 * w[3] * x[2];
  y[1] = 0.5 * w[1] * x[0] + w[2] * x[1] + 0.5 * w[4] * x[2];
  y[2] = 0.5 * w[3] * x[0] + 0.5 * w[4] * x[1] + w[5] * x[2];
}

struct GradWeights {
  double op[10];  // weights of dS, dD_A (3), dQ_A (6)
  double p, hij, hs, dlog, vb[3], wb[6];
};

__device__ __forceinline__ void grad_weights(const KsBatch& B, const KsPairs& P,
                                             const KsGradIn& G, const PairCtx& c, int pair,
                                             GradWeights& w) {
  const long long idx = c.base + (long long)c.mu * c.nao + c.nu;
  const double p = G.P[idx], x = G.X[idx];
  const int ao0 = B.mol_ao[c.m];
  const double rr = sqrt(sqrt(c.r2) / (B.atom_rad[c.A] + B.atom_rad[c.Bt]));
  const double shi = B.sh_shpoly[c.si], shj = B.sh_shpoly[c.sj];
  const double pi = 1.0 + shi * rr, pj = 1.0 + shj * rr;
  w.hs = P.hscale[pair] * pi * pj;
  w.hij = 0.5 * (B.sh_se[c.si] + B.sh_se[c.sj]) * w.hs;
  w.dlog = (pi * shj + pj * shi) * 0.5 * rr / c.r2 / (pi * pj);
  w.p = p;
  double wa[6], vbv[3];
  cart_quad_weights(G.vqp + 6 * c.A, wa);
  cart_quad_weights(G.vqp + 6 * c.Bt, w.wb);
  quad_mv(w.wb, c.vec, vbv);
  double vdpb_v = 0.0, vvv = 0.0;
#pragma unroll
  for (int d = 0; d < 3; ++d) {
    w.vb[d] = G.vdp[3 * c.Bt + d];
    vdpb_v += w.vb[d] * c.vec[d];
    vvv += c.vec[d] * vbv[d];
    w.op[1 + d] = -p * (G.vdp[3 * c.A + d] + w.vb[d] + 2.0 * vbv[d]);
  }
  const double wsum = -p * (G.vao[ao0 + c.mu] + G.vao[ao0 + c.nu]) + 2.0 * p * w.hij - 2.0 * x;
  w.op[0] = wsum - p * (vdpb_v + vvv);
#pragma unroll
  for (int k = 0; k < 6; ++k) w.op[4 + k] = -p * (wa[k] + w.wb[k]);
}

// sum over primitive and cartesian pairs: acc[e] = sum op . d(components)/dA_e, val = S, D_A
template <int LI, int LJ>
__device__ void block_grad(const KsBatch& B, const PairCtx& c, int mi, int mj,
                           const double op[10], double acc[3], double val[4]) {
  double wi[Ang<LI>::ncart], wj[Ang<LJ>::ncart];
  sph_weights<LI>(mi, wi);
  sph_weights<LJ>(mj, wj);
  const int cgi = B.sh_cg[c.si], cgj = B.sh_cg[c.sj];
  const int npi = B.cg_nprim[cgi], npj = B.cg_nprim[cgj];
  for (int ip = 0; ip < npi; ++ip) {
    for (int jp = 0; jp < npj; ++jp) {
      PrimPair pp;
      if (!prim_pair(B, cgi, ip, cgj, jp, c.r2, pp)) continue;
      Tab1D<LI, LJ> t[3];
#pragma unroll
      for (int d = 0; d < 3; ++d) t[d].build(-c.vec[d] * pp.aj * pp.oab, c.vec[d], 0.5 * pp.oab);
      static_for<Ang<LI>::ncart>([&](auto A) {
        static_for<Ang<LJ>::ncart>([&](auto Bc) {
          constexpr int a = decltype(A)::value, b = decltype(Bc)::value;
          const double tw = wi[a] * wj[b] * pp.cc;
          if (tw == 0.0) return;
          double m[3][3], g[3][3], v[10];
          moments<LI, LJ, cart_e(LI, a, 0), cart_e(LJ, b, 0)>(t[0], m[0]);
          moments<LI, LJ, cart_e(LI, a, 1), cart_e(LJ, b, 1)>(t[1], m[1]);
          moments<LI, LJ, cart_e(LI, a, 2), cart_e(LJ, b, 2)>(t[2], m[2]);
          dmoments<LI, LJ, cart_e(LI, a, 0), cart_e(LJ, b, 0)>(t[0], pp.ai, g[0]);
          dmoments<LI, LJ, cart_e(LI, a, 1), cart_e(LJ, b, 1)>(t[1], pp.ai, g[1]);
          dmoments<LI, LJ, cart_e(LI, a, 2), cart_e(LJ, b, 2)>(t[2], pp.ai, g[2]);
          assemble(m, v);
#pragma unroll
          for (int k = 0; k < 4; ++k) val[k] += tw * v[k];
#pragma unroll
          for (int e = 0; e < 3; ++e) {
            double mm[3][3];
#pragma unroll
            for (int d = 0; d < 3; ++d)
#pragma unroll
              for (int k = 0; k < 3; ++k) mm[d][k] = d == e ? g[d][k] : m[d][k];
            assemble(mm, v);
            double sum = 0.0;
#pragma unroll
            for (int k = 0; k < 10; ++k) sum += op[k] * v[k];
            acc[e] += tw * sum;
          }
        });
      });
    }
  }
}

template <int LI, int LJ>
__global__ void __launch_bounds__(128) grad_kernel(KsBatch B, KsPairs P, KsGradIn G, KsGradOut O) {
  constexpr int NSI = Ang<LI>::nsph, NSJ = Ang<LJ>::nsph;
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)P.n * NSI * NSJ) return;
  const int pair = (int)(t % P.n), mij = (int)(t / P.n), mi = mij / NSJ, mj = mij % NSJ;
  PairCtx c;
  if (!pair_ctx(B, P, pair, mi, mj, c) || c.r2 <= 0.0) return;
  GradWeights w;
  grad_weights(B, P, G, c, pair, w);
  double acc[3] = {0.0, 0.0, 0.0}, val[4] = {0.0, 0.0, 0.0, 0.0};
  block_grad<LI, LJ>(B, c, mi, mj, w.op, acc, val);
  const double s = val[0];
  double vbv[3], vbd[3];
  quad_mv(w.wb, c.vec, vbv);
  quad_mv(w.wb, val + 1, vbd);
#pragma unroll
  for (int e = 0; e < 3; ++e) {
    const double dg = acc[e] + 2.0 * w.hij * w.p * s * w.dlog * c.vec[e] -
                      w.p * (w.vb[e] * s + 2.0 * vbd[e] + 2.0 * vbv[e] * s);
    atomicAdd(O.grad + 3 * c.A + e, dg);
    atomicAdd(O.grad + 3 * c.Bt + e, -dg);
  }
  const double ps = w.p * s * w.hs;
  atomicAdd(O.dedcn + c.A, B.sh_dsedcn[c.si] * ps);
  atomicAdd(O.dedcn + c.Bt, B.sh_dsedcn[c.sj] * ps);
}

// On-site dE/dCN: dsedcn of each shell times the trace of its diagonal density block.
__global__ void onsite_dedcn_kernel(KsBatch B, KsGradIn G, KsGradOut O) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= B.nsh) return;
  const int atom = B.sh_atom[s], m = B.atom_mol[atom], nao = B.mol_nao[m];
  const int ao = B.sh_ao[s];  // the shell ends where the molecule's next shell starts
  const int next = (s + 1 < B.nsh && B.atom_mol[B.sh_atom[s + 1]] == m) ? B.sh_ao[s + 1] : nao;
  double tr = 0.0;
  for (int mu = ao; mu < next; ++mu) tr += G.P[B.mol_mat[m] + (long long)mu * nao + mu];
  atomicAdd(O.dedcn + atom, B.sh_dsedcn[s] * tr);
}

template <int LI, int LJ>
cudaError_t launch_grad(const KsBatch& b, const KsPairs& p, const KsGradIn& g,
                        const KsGradOut& o, cudaStream_t s) {
  const long long n = (long long)p.n * Ang<LI>::nsph * Ang<LJ>::nsph;
  if (n == 0) return cudaSuccess;
  grad_kernel<LI, LJ><<<(int)((n + 127) / 128), 128, 0, s>>>(b, p, g, o);
  return cudaGetLastError();
}

using GradFn = cudaError_t (*)(const KsBatch&, const KsPairs&, const KsGradIn&,
                               const KsGradOut&, cudaStream_t);
const GradFn kGrad[3][3] = {
    {launch_grad<0, 0>, launch_grad<0, 1>, launch_grad<0, 2>},
    {launch_grad<1, 0>, launch_grad<1, 1>, launch_grad<1, 2>},
    {launch_grad<2, 0>, launch_grad<2, 1>, launch_grad<2, 2>},
};

}  // namespace

extern "C" int ksxtb_h0_gradient(const KsBatch* b, const KsPairs* lists, int nlists,
                                 const KsGradIn* in, const KsGradOut* out, void* stream) {
  const cudaStream_t s = (cudaStream_t)stream;
  for (int i = 0; i < nlists; ++i) {
    const KsPairs& p = lists[i];
    if (p.onsite) continue;
    if (p.li < 0 || p.li > 2 || p.lj < 0 || p.lj > 2) return (int)cudaErrorInvalidValue;
    const cudaError_t e = kGrad[p.li][p.lj](*b, p, *in, *out, s);
    if (e != cudaSuccess) return (int)e;
  }
  if (b->nsh > 0) onsite_dedcn_kernel<<<(b->nsh + 127) / 128, 128, 0, s>>>(*b, *in, *out);
  return (int)cudaGetLastError();
}
