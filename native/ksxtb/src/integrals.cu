// Batched GFN2-xTB one-electron integrals: overlap, dipole and quadrupole moments, H0, and
// the H0 gradient contraction of tblite (xtb/h0.f90, integral/multipole.f90).
//
// Mapping: one thread per (shell pair, spherical AO pair). Thread t of a class kernel takes
// pair t % n and AO pair t / n, so a warp shares one AO pair (mi, mj) and every branch on it is
// warp-uniform; the cartesian loops are unrolled at compile time, so the 1D tables stay in
// registers. The 1D integrals use the Obara-Saika recursion over the product Gaussian and the
// transfer relation (x - B) = (x - A) + (A - B).
#include <cuda_runtime.h>

#include <type_traits>
#include <utility>

#include "ksxtb.h"

namespace {

// Compile-time loop: f(std::integral_constant<int, i>) for i = 0..N-1.
template <typename F, int... I>
__device__ __forceinline__ void static_for_impl(F& f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>{}), ...);
}
template <int N, typename F>
__device__ __forceinline__ void static_for(F&& f) {
  static_for_impl(f, std::make_integer_sequence<int, N>{});
}

constexpr double kPi32 = 5.5683279968317078;  // pi^(3/2)
constexpr double kIntCut = 25.0;              // tblite's integral_cutoff for every accuracy
constexpr double kS3 = 1.7320508075688772;

template <int L>
struct Ang {
  static constexpr int nsph = 2 * L + 1;
  static constexpr int ncart = (L + 1) * (L + 2) / 2;
};

// Cartesian exponent of component a of an l shell along d, tblite's order:
// p = y, z, x; d = xx, yy, zz, xy, xz, yz.
__host__ __device__ constexpr int cart_e(int l, int a, int d) {
  constexpr int p[3][3] = {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}};
  constexpr int dd[6][3] = {{2, 0, 0}, {0, 2, 0}, {0, 0, 2}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
  return l == 0 ? 0 : (l == 1 ? p[a][d] : dd[a][d]);
}

// Weights of the cartesian components in spherical function m (tblite's dtrafo).
template <int L>
__device__ __forceinline__ void sph_weights(int m, double w[Ang<L>::ncart]) {
#pragma unroll
  for (int a = 0; a < Ang<L>::ncart; ++a) w[a] = (L < 2 && a == m) ? 1.0 : 0.0;
  if (L < 2) return;
  switch (m) {
    case 0: w[3] = kS3; break;  // xy
    case 1: w[5] = kS3; break;  // yz
    case 2: w[0] = -0.5; w[1] = -0.5; w[2] = 1.0; break;  // z2
    case 3: w[4] = kS3; break;  // xz
    default: w[0] = 0.5 * kS3; w[1] = -0.5 * kS3; break;  // x2-y2
  }
}

// 1D integrals I(n, b) = <(x-A)^n (x-B)^b> over the product Gaussian, normalized by its
// 1D weight, for n = 0..LI+3 and b = 0..LJ.
template <int LI, int LJ>
struct Tab1D {
  static constexpr int NA = LI + 4, NB = LJ + 1, N = LI + 3 + LJ;
  double v[NA][NB];

  __device__ __forceinline__ void build(double pa, double ab, double half_inv_eab) {
    double w[NB][N + 1];
    w[0][0] = 1.0;
    w[0][1] = pa;
#pragma unroll
    for (int n = 1; n < N; ++n) w[0][n + 1] = pa * w[0][n] + n * half_inv_eab * w[0][n - 1];
#pragma unroll
    for (int b = 1; b < NB; ++b) {
#pragma unroll
      for (int n = 0; n <= N - b; ++n) w[b][n] = w[b - 1][n + 1] + ab * w[b - 1][n];
    }
#pragma unroll
    for (int n = 0; n < NA; ++n) {
#pragma unroll
      for (int b = 0; b < NB; ++b) v[n][b] = w[b][n];
    }
  }
};

struct PrimPair {
  double cc, ai, aj, oab;
};

__device__ __forceinline__ bool prim_pair(const KsBatch& B, int cgi, int ip, int cgj, int jp,
                                          double r2, PrimPair& pp) {
  const double ai = __ldg(B.cg_alpha + cgi * 6 + ip), aj = __ldg(B.cg_alpha + cgj * 6 + jp);
  const double eab = ai + aj, oab = 1.0 / eab, est = ai * aj * r2 * oab;
  if (est > kIntCut) return false;
  pp.cc = __ldg(B.cg_coeff + cgi * 6 + ip) * __ldg(B.cg_coeff + cgj * 6 + jp) * exp(-est) *
          kPi32 * oab * sqrt(oab);
  pp.ai = ai;
  pp.aj = aj;
  pp.oab = oab;
  return true;
}

// Moments k = 0..2 about A of one cartesian pair along one dimension.
template <int LI, int LJ, int EA, int EB>
__device__ __forceinline__ void moments(const Tab1D<LI, LJ>& t, double m[3]) {
  m[0] = t.v[EA][EB];
  m[1] = t.v[EA + 1][EB];
  m[2] = t.v[EA + 2][EB];
}

// d/dA of those moments, the operator centre moving with A: the function's derivative
// -EA (x-A)^(EA-1) + 2 ai (x-A)^(EA+1), minus k (x-A)^(k-1) of the operator.
template <int LI, int LJ, int EA, int EB>
__device__ __forceinline__ void dmoments(const Tab1D<LI, LJ>& t, double ai, double g[3]) {
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    g[k] = 2.0 * ai * t.v[EA + 1 + k][EB];
    if constexpr (EA > 0) g[k] -= EA * t.v[EA - 1 + k][EB];
  }
  g[1] -= t.v[EA][EB];
  g[2] -= 2.0 * t.v[EA + 1][EB];
}

// The 10 operator components (S, D x y z, Q xx xy yy xz yz zz) from per-dimension factors.
__device__ __forceinline__ void assemble(const double (&m)[3][3], double out[10]) {
  out[0] = m[0][0] * m[1][0] * m[2][0];
  out[1] = m[0][1] * m[1][0] * m[2][0];
  out[2] = m[0][0] * m[1][1] * m[2][0];
  out[3] = m[0][0] * m[1][0] * m[2][1];
  out[4] = m[0][2] * m[1][0] * m[2][0];
  out[5] = m[0][1] * m[1][1] * m[2][0];
  out[6] = m[0][0] * m[1][2] * m[2][0];
  out[7] = m[0][1] * m[1][0] * m[2][1];
  out[8] = m[0][0] * m[1][1] * m[2][1];
  out[9] = m[0][0] * m[1][0] * m[2][2];
}

__device__ __forceinline__ void traceless(double q[6]) {
  const double tr = 0.5 * (q[0] + q[2] + q[5]);
#pragma unroll
  for (int c = 0; c < 6; ++c) q[c] *= 1.5;
  q[0] -= tr;
  q[2] -= tr;
  q[5] -= tr;
}

// Cartesian second moments about B from those about A: (r-B) = (r-A) + v, v = A - B.
__device__ __forceinline__ void shift_to_b(const double v[3], double s, const double d[3],
                                           const double qa[6], double qb[6]) {
  qb[0] = qa[0] + 2 * v[0] * d[0] + v[0] * v[0] * s;
  qb[1] = qa[1] + v[0] * d[1] + v[1] * d[0] + v[0] * v[1] * s;
  qb[2] = qa[2] + 2 * v[1] * d[1] + v[1] * v[1] * s;
  qb[3] = qa[3] + v[0] * d[2] + v[2] * d[0] + v[0] * v[2] * s;
  qb[4] = qa[4] + v[1] * d[2] + v[2] * d[1] + v[1] * v[2] * s;
  qb[5] = qa[5] + 2 * v[2] * d[2] + v[2] * v[2] * s;
}

struct PairCtx {
  int si, sj, A, Bt, m, mu, nu, nao;
  long long base;
  double vec[3], r2;
};

__device__ __forceinline__ bool pair_ctx(const KsBatch& B, const KsPairs& P, int pair, int mi,
                                         int mj, PairCtx& c) {
  c.si = P.si[pair];
  c.sj = P.sj[pair];
  c.A = B.sh_atom[c.si];
  c.Bt = B.sh_atom[c.sj];
  c.m = B.atom_mol[c.A];
  c.nao = B.mol_nao[c.m];
  c.base = B.mol_mat[c.m];
  c.mu = B.sh_ao[c.si] + mi;
  c.nu = B.sh_ao[c.sj] + mj;
  c.r2 = 0.0;
#pragma unroll
  for (int d = 0; d < 3; ++d) {
    c.vec[d] = B.xyz[3 * c.A + d] - B.xyz[3 * c.Bt + d];
    c.r2 += c.vec[d] * c.vec[d];
  }
  return P.onsite || c.r2 <= B.mol_cut2[c.m];
}

// Sum over primitive pairs and cartesian pairs of the 10 components for AO pair (mi, mj).
template <int LI, int LJ>
__device__ void block_values(const KsBatch& B, const PairCtx& c, int mi, int mj, double acc[10]) {
  double wi[Ang<LI>::ncart], wj[Ang<LJ>::ncart];
  sph_weights<LI>(mi, wi);
  sph_weights<LJ>(mj, wj);
  const int cgi = B.sh_cg[c.si], cgj = B.sh_cg[c.sj];
  const int npi = B.cg_nprim[cgi], npj = B.cg_nprim[cgj];
#pragma unroll
  for (int k = 0; k < 10; ++k) acc[k] = 0.0;
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
          double m[3][3], v[10];
          moments<LI, LJ, cart_e(LI, a, 0), cart_e(LJ, b, 0)>(t[0], m[0]);
          moments<LI, LJ, cart_e(LI, a, 1), cart_e(LJ, b, 1)>(t[1], m[1]);
          moments<LI, LJ, cart_e(LI, a, 2), cart_e(LJ, b, 2)>(t[2], m[2]);
          assemble(m, v);
#pragma unroll
          for (int k = 0; k < 10; ++k) acc[k] += tw * v[k];
        });
      });
    }
  }
}

__device__ __forceinline__ void put(double* M, int ncomp, const PairCtx& c, int comp, int row,
                                    int col, double val) {
  M[ncomp * c.base + (long long)comp * c.nao * c.nao + (long long)row * c.nao + col] = val;
}

template <int LI, int LJ, bool ONSITE>
__global__ void __launch_bounds__(128) forward_kernel(KsBatch B, KsPairs P, KsFwdOut O) {
  constexpr int NSI = Ang<LI>::nsph, NSJ = Ang<LJ>::nsph;
  const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)P.n * NSI * NSJ) return;
  const int pair = (int)(t % P.n), mij = (int)(t / P.n), mi = mij / NSJ, mj = mij % NSJ;
  PairCtx c;
  double v[10];
  const bool inside = pair_ctx(B, P, pair, mi, mj, c);
  if (inside) {
    block_values<LI, LJ>(B, c, mi, mj, v);
  } else {
#pragma unroll
    for (int k = 0; k < 10; ++k) v[k] = 0.0;
  }
  double qa[6], qb[6], db[3];
#pragma unroll
  for (int k = 0; k < 6; ++k) qa[k] = v[4 + k];
  const double s = v[0];
  double hij;
  if (ONSITE) {
    hij = 0.5 * (B.sh_se[c.si] + B.sh_se[c.sj]);
  } else {
    const double rr = sqrt(sqrt(c.r2) / (B.atom_rad[c.A] + B.atom_rad[c.Bt]));
    hij = 0.5 * (B.sh_se[c.si] + B.sh_se[c.sj]) * P.hscale[pair] *
          (1.0 + B.sh_shpoly[c.si] * rr) * (1.0 + B.sh_shpoly[c.sj] * rr);
    shift_to_b(c.vec, s, v + 1, qa, qb);
#pragma unroll
    for (int d = 0; d < 3; ++d) db[d] = v[1 + d] + c.vec[d] * s;
    traceless(qb);
  }
  traceless(qa);
  put(O.S, 1, c, 0, c.mu, c.nu, s);
  put(O.H0, 1, c, 0, c.mu, c.nu, s * hij);
  if (ONSITE) {  // <mu|(r - A)|nu>, both on A
#pragma unroll
    for (int d = 0; d < 3; ++d) put(O.D, 3, c, d, c.mu, c.nu, v[1 + d]);
#pragma unroll
    for (int k = 0; k < 6; ++k) put(O.Q, 6, c, k, c.mu, c.nu, qa[k]);
    return;
  }
  put(O.S, 1, c, 0, c.nu, c.mu, s);
  put(O.H0, 1, c, 0, c.nu, c.mu, s * hij);
#pragma unroll
  for (int d = 0; d < 3; ++d) {
    put(O.D, 3, c, d, c.nu, c.mu, v[1 + d]);  // column mu on A: centred on A
    put(O.D, 3, c, d, c.mu, c.nu, db[d]);     // column nu on B: centred on B
  }
#pragma unroll
  for (int k = 0; k < 6; ++k) {
    put(O.Q, 6, c, k, c.nu, c.mu, qa[k]);
    put(O.Q, 6, c, k, c.mu, c.nu, qb[k]);
  }
}

}  // namespace

#include "gradient.cuh"

namespace {

template <int LI, int LJ>
cudaError_t launch_forward(const KsBatch& b, const KsPairs& p, const KsFwdOut& o,
                           cudaStream_t s) {
  const long long n = (long long)p.n * Ang<LI>::nsph * Ang<LJ>::nsph;
  if (n == 0) return cudaSuccess;
  const int grid = (int)((n + 127) / 128);
  if (p.onsite)
    forward_kernel<LI, LJ, true><<<grid, 128, 0, s>>>(b, p, o);
  else
    forward_kernel<LI, LJ, false><<<grid, 128, 0, s>>>(b, p, o);
  return cudaGetLastError();
}

using FwdFn = cudaError_t (*)(const KsBatch&, const KsPairs&, const KsFwdOut&, cudaStream_t);
const FwdFn kForward[3][3] = {
    {launch_forward<0, 0>, launch_forward<0, 1>, launch_forward<0, 2>},
    {launch_forward<1, 0>, launch_forward<1, 1>, launch_forward<1, 2>},
    {launch_forward<2, 0>, launch_forward<2, 1>, launch_forward<2, 2>},
};

}  // namespace

extern "C" int ksxtb_forward(const KsBatch* b, const KsPairs* lists, int nlists,
                             const KsFwdOut* out, void* stream) {
  for (int i = 0; i < nlists; ++i) {
    const KsPairs& p = lists[i];
    if (p.li < 0 || p.li > 2 || p.lj < 0 || p.lj > 2) return (int)cudaErrorInvalidValue;
    const cudaError_t e = kForward[p.li][p.lj](*b, p, *out, (cudaStream_t)stream);
    if (e != cudaSuccess) return (int)e;
  }
  return (int)cudaSuccess;
}
