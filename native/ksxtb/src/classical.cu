// ksxtb stage 2: the GFN2 terms beyond the one-electron integrals, batched over molecules.
//
// Repulsion, ES2 (shell-resolved), ES3, AES (damped multipoles with CN-dependent radii), D4
// (two-body with the SCF charges, ATM with zero-charge C6), ALPB (P16 Born kernel and the
// shape term) and CDS (smoothed SASA on the 230-point Lebedev grid), following tblite 0.7.0 as
// `ksmech.qm.gpu.terms` writes it. All in fp64.
//
// No derivative tensor is stored. Each pass accumulates per-atom adjoints (dE/dCN of both CNs,
// dE/dmultipole radius, dE/dBorn radius) with atomics; one chain pass over the pairs turns
// them into gradient. One thread per atom pair (one molecule's pairs are contiguous), per atom,
// per (atom, grid point) or per molecule.
#include <cuda_runtime.h>
#include <math.h>

#include "ksxtb.h"

namespace {

constexpr int kThreads = 128;
constexpr double kSqrtPi = 1.7724538509055160273;
constexpr double kAatoau = 1.0 / 0.529177210903;
constexpr double kCnDexpCut2 = 25.0 * 25.0;      // mctc-lib dexp default
constexpr double kCnD4Cut2 = 30.0 * 30.0;        // dftd4 realspace_cutoff%cn
constexpr double kRepCut2 = 25.0 * 25.0;         // tb_repulsion
constexpr double kDisp2Cut = 50.0, kDisp3Cut = 25.0, kDispWidth = 0.05;
constexpr double kBornCut2 = (35.0 * kAatoau) * (35.0 * kAatoau);  // born.f90 lrcut_default
constexpr double kZetaP16 = 1.028;
constexpr double kObc0 = 1.0, kObc1 = 0.8, kObc2 = 4.85;
constexpr double kK4 = 4.10451, kK5 = 19.08857, kK6 = 2.0 * 11.28174 * 11.28174;
constexpr double kD4Kcn = 7.5;
constexpr double kAtmAlpha = 16.0;
constexpr double kTolSesp = 1e-6;

__device__ __forceinline__ double at(const KsClassical& c, int a, int col) {
  return c.atab[(long long)a * KA_COLS + col];
}
__device__ __forceinline__ int it(const KsClassical& c, int a, int col) {
  return c.itab[a * KI_COLS + col];
}
__device__ __forceinline__ void add3(double* g, int a, const double* v, double s) {
  atomicAdd(g + 3 * a + 0, s * v[0]);
  atomicAdd(g + 3 * a + 1, s * v[1]);
  atomicAdd(g + 3 * a + 2, s * v[2]);
}
__device__ __forceinline__ double dot3(const double* a, const double* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
__device__ __forceinline__ double pair_vec(const KsClassical& c, int i, int j, double* v) {
  v[0] = c.xyz[3 * i + 0] - c.xyz[3 * j + 0];
  v[1] = c.xyz[3 * i + 1] - c.xyz[3 * j + 1];
  v[2] = c.xyz[3 * i + 2] - c.xyz[3 * j + 2];
  return dot3(v, v);
}

// logistic count of mctc-lib's dexp CN and its r-derivative
__device__ __forceinline__ void exp_count(double k, double r, double r0, double* f, double* df) {
  double e = exp(-k * (r0 / r - 1.0)), cf = 1.0 / (1.0 + e);
  *f = cf;
  *df = -k * r0 * e * cf * cf / (r * r);
}
__device__ __forceinline__ void dexp_count(const KsClassical& c, int i, int j, double r,
                                           double* f, double* df) {
  double rc = at(c, i, KA_RCOV) + at(c, j, KA_RCOV), f1, d1, f2, d2;
  exp_count(10.0, r, rc, &f1, &d1);
  exp_count(20.0, r, rc + 2.0, &f2, &d2);
  *f = f1 * f2;
  *df = d1 * f2 + f1 * d2;
}
__device__ __forceinline__ void d4_count(const KsClassical& c, int i, int j, double r, double* f,
                                         double* df) {
  double rc = at(c, i, KA_RCOV4) + at(c, j, KA_RCOV4);
  double den = fabs(at(c, i, KA_EN4) - at(c, j, KA_EN4)) + kK5;
  den = kK4 * exp(-den * den / kK6);
  double x = -kD4Kcn * (r - rc) / rc;
  *f = den * 0.5 * (1.0 + erf(x));
  *df = -den * kD4Kcn / rc / kSqrtPi * exp(-x * x);
}

// descreening integral of one neighbour (born.f90 compute_psi) and (dg/dr) / r
__device__ double born_g(double r, double rho, double rvdw, double* dg) {
  double r1 = 1.0 / r, ap = r + rho, am = r - rho;
  if (r >= rvdw + rho) {
    double ab = ap * am, rhab = rho / ab, lnab = 0.5 * log(am / ap) * r1;
    *dg = -2.0 * rhab / ab + (rhab - lnab) * r1 * r1;
    return rhab + lnab;
  }
  if (ap > rvdw) {
    double r12 = 0.5 * r1, r24 = r12 * r12, rh1 = 1.0 / rvdw, rhr1 = 1.0 / ap;
    double aprh1 = ap * rh1, lnab = log(aprh1);
    *dg = (rhr1 * rhr1 * (1.0 - 0.25 * am * r1 * (1.0 + aprh1 * aprh1)) +
           rho * r24 * (rhr1 - rh1 * aprh1) + r12 * (r1 * lnab - rhr1)) * r1;
    return rh1 - rhr1 + r12 * (0.5 * am * (rhr1 - rh1 * aprh1) - lnab);
  }
  *dg = 0.0;
  return 0.0;
}

// dftd4's smooth_cutoff and its r-derivative
__device__ __forceinline__ double smooth(double r, double cut, double width, double* dsw) {
  if (r <= cut - width) {
    *dsw = 0.0;
    return 1.0;
  }
  if (r >= cut) {
    *dsw = 0.0;
    return 0.0;
  }
  double x = (cut - r) / width;
  *dsw = -30.0 * x * x * (1.0 - x) * (1.0 - x) / width;
  return x * x * x * (10.0 + x * (-15.0 + 6.0 * x));
}

__device__ __forceinline__ long long pair_index(const KsClassical& c, int m, int i, int j) {
  int il = i - c.mol_at0[m], jl = j - c.mol_at0[m];
  return c.mol_pair0[m] + (long long)il * (il - 1) / 2 + jl;
}

// ------------------------------------------------------------------------------ geometry

__global__ void __launch_bounds__(kThreads) cn_psi_kernel(KsClassical c, KsWork w) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= c.npair) return;
  int i = c.pi[p], j = c.pj[p];
  double v[3], r2 = pair_vec(c, i, j, v);
  if (r2 < 1e-12) return;
  double r = sqrt(r2), f, df;
  if (r2 <= kCnDexpCut2) {
    dexp_count(c, i, j, r, &f, &df);
    atomicAdd(w.cn + i, f);
    atomicAdd(w.cn + j, f);
  }
  if (r2 <= kCnD4Cut2) {
    d4_count(c, i, j, r, &f, &df);
    atomicAdd(w.cn4 + i, f);
    atomicAdd(w.cn4 + j, f);
  }
  if (c.solvent && r2 <= kBornCut2) {
    double dg;
    atomicAdd(w.psi + i, born_g(r, at(c, j, KA_RHO), at(c, i, KA_RVDW), &dg));
    atomicAdd(w.psi + j, born_g(r, at(c, i, KA_RHO), at(c, j, KA_RVDW), &dg));
  }
}

__global__ void __launch_bounds__(kThreads) atom_geometry_kernel(KsClassical c, KsWork w) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= c.nat) return;
  double rad = at(c, a, KA_MP_RAD), kexp = c.sc[KS_MP_KEXP];
  double t1 = exp(-kexp * (w.cn[a] - at(c, a, KA_MP_VCN) - c.sc[KS_MP_SHIFT]));
  double t2 = (c.sc[KS_MP_RMAX] - rad) / (1.0 + t1);
  w.mrad[a] = rad + t2;
  w.dmrdcn[a] = t2 * kexp * t1 / (1.0 + t1);
  if (!c.solvent) return;
  double svdw = at(c, a, KA_SVDW), rvdw = at(c, a, KA_RVDW), c1 = c.sc[KS_BORN_C1];
  double br = w.psi[a] * 0.5 * svdw;
  double arg2 = br * (kObc2 * br - kObc1);
  double arg = br * (kObc0 + arg2);
  double darg = 2.0 * arg2 + kObc0 + kObc2 * br * br;
  double th = tanh(arg), den = 1.0 / svdw - th / rvdw, ch = cosh(arg) * den;
  w.brad[a] = c1 / den;
  w.dbrdpsi[a] = c1 * 0.5 * svdw / rvdw * darg / (ch * ch);
}


// ------------------------------------------------------------------------------ per atom

__device__ __forceinline__ double zeta(double a, double c, double qref, double qmod,
                                       double* dz) {
  if (qmod < 0.0) {
    *dz = 0.0;
    return exp(a);
  }
  double e = exp(c * (1.0 - qref / qmod)), z = exp(a * (1.0 - e));
  *dz = -a * c * e * z * qref / (qmod * qmod);
  return z;
}

// dftd4's weight_references for one atom: gw, dgw/dCN, dgw/dq of its references
__device__ void d4_weights(const KsClassical& c, int a, double cn, double q, double* gw,
                           double* dgwdcn, double* dgwdq) {
  const int nref = it(c, a, KI_NREF);
  const double wf = c.sc[KS_D4_WF];
  double expw[KS_NREF], expd[KS_NREF], norm = 0.0, dnorm = 0.0, maxcn = -1e300;
  for (int r = 0; r < nref; ++r) {
    double refcn = at(c, a, KA_REFCN + r), d = refcn - cn;
    expw[r] = expd[r] = 0.0;
    for (int k = 1; k <= it(c, a, KI_NGW + r); ++k) {
      double g = exp(-k * wf * d * d);
      expw[r] += g;
      expd[r] += 2.0 * k * wf * d * g;
    }
    norm += expw[r];
    dnorm += expd[r];
    maxcn = fmax(maxcn, refcn);
  }
  bool degenerate = !(norm > 1.4916681462400413e-154);  // sqrt(tiny)
  double inv = degenerate ? 0.0 : 1.0 / norm;
  double zi = at(c, a, KA_D4_ZI), gi = at(c, a, KA_D4_GI), ga = c.sc[KS_D4_GA];
  for (int r = 0; r < nref; ++r) {
    double gwk = expw[r] * inv, dgwk = inv * (expd[r] - expw[r] * dnorm * inv);
    if (degenerate || !isfinite(gwk)) {
      gwk = fabs(maxcn - at(c, a, KA_REFCN + r)) < 1e-12 ? 1.0 : 0.0;
      dgwk = 0.0;
    }
    if (!isfinite(dgwk)) dgwk = 0.0;
    double dz, z = zeta(ga, gi, at(c, a, KA_REFQ + r) + zi, q + zi, &dz);
    gw[r] = gwk * z;
    dgwdcn[r] = dgwk * z;
    dgwdq[r] = gwk * dz;
  }
  for (int r = nref; r < KS_NREF; ++r) gw[r] = dgwdcn[r] = dgwdq[r] = 0.0;
}

__global__ void __launch_bounds__(kThreads) atom_scf_kernel(KsClassical c, KsWork w, KsScf s,
                                                            KsClsOut o) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= c.nat) return;
  double tmp[KS_NREF];
  d4_weights(c, a, w.cn4[a], s.qat[a], w.gw + KS_NREF * a, w.dgwdcn + KS_NREF * a,
             w.dgwdq + KS_NREF * a);
  d4_weights(c, a, w.cn4[a], 0.0, w.gw0 + KS_NREF * a, w.dgw0dcn + KS_NREF * a, tmp);
  double* ea = w.eatom + KE_N * a;
  // ES3 and the on-site ES2 block, eta_ab = (eta_a + eta_b) / 2
  int sh0 = it(c, a, KI_SH0), nsh = it(c, a, KI_NSH);
  double e2 = 0.0, e3 = 0.0;
  for (int x = sh0; x < sh0 + nsh; ++x) {
    double qx = s.qsh[x], v = qx * qx * c.sh_hd[x];
    e3 += qx * v / 3.0;
    for (int y = sh0; y < sh0 + nsh; ++y) {
      double g = 0.5 * (c.sh_eta[x] + c.sh_eta[y]);
      v += g * s.qsh[y];
      e2 += 0.5 * qx * g * s.qsh[y];
    }
    atomicAdd(o.vsh + x, v);
  }
  atomicAdd(ea + KE_ES2, e2);
  atomicAdd(ea + KE_ES3, e3);
  // AES self kernels: dk |dp|^2 + qk sum scale qp^2
  const double scale[6] = {1, 2, 1, 2, 2, 1};
  double dk = at(c, a, KA_DKERNEL), qk = at(c, a, KA_QKERNEL), eself = 0.0;
  for (int k = 0; k < 3; ++k) {
    double d = s.dp[3 * a + k];
    eself += dk * d * d;
    atomicAdd(o.vdp + 3 * a + k, 2.0 * dk * d);
  }
  for (int k = 0; k < 6; ++k) {
    double d = s.qp[6 * a + k];
    eself += qk * scale[k] * d * d;
    atomicAdd(o.vqp + 6 * a + k, 2.0 * qk * scale[k] * d);
  }
  atomicAdd(ea + KE_AES, eself);
  if (!c.solvent) return;
  // ALPB diagonal keps q^2 / (2 br)
  double keps = c.sc[KS_KEPS], q = s.qat[a], br = w.brad[a];
  atomicAdd(ea + KE_ALPB, 0.5 * keps * q * q / br);
  atomicAdd(o.vat + a, keps * q / br);
  atomicAdd(w.dedbr + a, -0.5 * keps * q * q / (br * br));
}

// ------------------------------------------------------------------------------ per pair
// v = x_i - x_j; every term adds half its pair energy to each atom and +dE/dv to i, -dE/dv to j

struct Pair {
  int i, j, m;
  double v[3], r2, r;
};

__device__ __forceinline__ void add_energy(const KsWork& w, const Pair& p, int term, double e) {
  atomicAdd(w.eatom + KE_N * p.i + term, 0.5 * e);
  atomicAdd(w.eatom + KE_N * p.j + term, 0.5 * e);
}
__device__ __forceinline__ void add_force(const KsClsOut& o, const Pair& p, const double* g) {
  add3(o.grad, p.i, g, 1.0);
  add3(o.grad, p.j, g, -1.0);
}

__device__ void rep_pair(const KsClassical& c, const KsWork& w, const KsClsOut& o,
                         const Pair& p) {
  if (p.r2 > kRepCut2) return;
  bool heavy = it(c, p.i, KI_Z) > 2 || it(c, p.j, KI_Z) > 2;
  double kexp = heavy ? c.sc[KS_REP_KEXP] : c.sc[KS_REP_KEXP_LIGHT], rexp = c.sc[KS_REP_REXP];
  double alpha = sqrt(at(c, p.i, KA_REP_ALPHA) * at(c, p.j, KA_REP_ALPHA));
  double r1k = pow(p.r, kexp);
  double e = at(c, p.i, KA_REP_ZEFF) * at(c, p.j, KA_REP_ZEFF) * exp(-alpha * r1k) / pow(p.r, rexp);
  double s = -(alpha * r1k * kexp + rexp) * e / p.r2, g[3] = {s * p.v[0], s * p.v[1], s * p.v[2]};
  add_energy(w, p, KE_REP, e);
  add_force(o, p, g);
}

__device__ void es2_pair(const KsClassical& c, const KsWork& w, const KsScf& s,
                         const KsClsOut& o, const Pair& p) {
  int si = it(c, p.i, KI_SH0), ni = it(c, p.i, KI_NSH);
  int sj = it(c, p.j, KI_SH0), nj = it(c, p.j, KI_NSH);
  double e = 0.0, gs = 0.0;
  for (int x = si; x < si + ni; ++x) {
    double vx = 0.0;
    for (int y = sj; y < sj + nj; ++y) {
      double eta = 0.5 * (c.sh_eta[x] + c.sh_eta[y]);
      double t = 1.0 / (p.r2 + 1.0 / (eta * eta)), gam = sqrt(t);  // gexp = 2
      double qq = s.qsh[x] * s.qsh[y];
      e += qq * gam;
      gs -= qq * gam * t;  // (d gamma / dr) / r = -(r^2 + eta^-2)^(-3/2)
      vx += gam * s.qsh[y];
      atomicAdd(o.vsh + y, gam * s.qsh[x]);
    }
    atomicAdd(o.vsh + x, vx);
  }
  double g[3] = {gs * p.v[0], gs * p.v[1], gs * p.v[2]};
  add_energy(w, p, KE_ES2, e);
  add_force(o, p, g);
}

// (qp as the symmetric matrix M) . v and v^T M v
__device__ __forceinline__ double quad_form(const double* q, const double* v, double* mv) {
  mv[0] = q[0] * v[0] + q[1] * v[1] + q[3] * v[2];
  mv[1] = q[1] * v[0] + q[2] * v[1] + q[4] * v[2];
  mv[2] = q[3] * v[0] + q[4] * v[1] + q[5] * v[2];
  return dot3(v, mv);
}

__device__ void aes_pair(const KsClassical& c, const KsWork& w, const KsScf& s,
                         const KsClsOut& o, const Pair& p) {
  const int i = p.i, j = p.j;
  const double *v = p.v, r = p.r, *dpi = s.dp + 3 * i, *dpj = s.dp + 3 * j;
  double qi = s.qat[i], qj = s.qat[j];
  double k3 = c.sc[KS_MP_DMP3], k5 = c.sc[KS_MP_DMP5];
  double u = 0.5 * (w.mrad[i] + w.mrad[j]) / r;
  double f3 = 1.0 / (1.0 + 6.0 * pow(u, k3)), f5 = 1.0 / (1.0 + 6.0 * pow(u, k5));
  double df3 = -6.0 * k3 * pow(u, k3 - 1.0) * f3 * f3, df5 = -6.0 * k5 * pow(u, k5 - 1.0) * f5 * f5;
  double r3 = r * p.r2, r4 = p.r2 * p.r2, r5 = r4 * r, r6 = r4 * p.r2;
  double g3 = f3 / r3, h3 = f5 / r3, g5 = f5 / r5;  // charge-dipole, dipole-dipole, quadrupole
  double dg3 = (-df3 * u - 3.0 * f3) / r4, dh3 = (-df5 * u - 3.0 * f5) / r4;  // d/dr
  double dg5 = (-df5 * u - 5.0 * f5) / r6;
  double mi[3], mj[3], qti = quad_form(s.qp + 6 * i, v, mi), qtj = quad_form(s.qp + 6 * j, v, mj);
  double dpiv = dot3(dpi, v), dpjv = dot3(dpj, v), dpij = dot3(dpi, dpj);
  double psd = dpjv * qi - dpiv * qj, pdd = dpiv * dpjv, psq = qtj * qi + qti * qj;
  add_energy(w, p, KE_AES, g3 * psd + h3 * dpij - 3.0 * g5 * pdd + g5 * psq);
  double rad = (dg3 * psd + dh3 * dpij - 3.0 * dg5 * pdd + dg5 * psq) / r, g[3];
  for (int k = 0; k < 3; ++k) {
    g[k] = rad * v[k] + g3 * (qi * dpj[k] - qj * dpi[k]) - 3.0 * g5 * (dpi[k] * dpjv + dpj[k] * dpiv)
         + 2.0 * g5 * (qi * mj[k] + qj * mi[k]);
  }
  add_force(o, p, g);
  double dedR = (df3 * psd + df5 * dpij) / r4 + df5 * (psq - 3.0 * pdd) / r6;  // d/dR, R = u r
  atomicAdd(w.dedcn + i, 0.5 * dedR);  // dE/dmrad until post_kernel multiplies by dmrad/dCN
  atomicAdd(w.dedcn + j, 0.5 * dedR);
  atomicAdd(o.vat + i, g3 * dpjv + g5 * qtj);
  atomicAdd(o.vat + j, -g3 * dpiv + g5 * qti);
  for (int k = 0; k < 3; ++k) {
    atomicAdd(o.vdp + 3 * i + k, -g3 * qj * v[k] + h3 * dpj[k] - 3.0 * g5 * dpjv * v[k]);
    atomicAdd(o.vdp + 3 * j + k, g3 * qi * v[k] + h3 * dpi[k] - 3.0 * g5 * dpiv * v[k]);
  }
  const double tc[6] = {v[0] * v[0], 2 * v[0] * v[1], v[1] * v[1],
                        2 * v[0] * v[2], 2 * v[1] * v[2], v[2] * v[2]};
  for (int k = 0; k < 6; ++k) {
    atomicAdd(o.vqp + 6 * i + k, g5 * qj * tc[k]);
    atomicAdd(o.vqp + 6 * j + k, g5 * qi * tc[k]);
  }
}

// sum_ab x_a C_ab y_b over the species pair's reference C6 block
__device__ __forceinline__ double c6_contract(const double* ref, const double* x, int na,
                                              const double* y, int nb) {
  double acc = 0.0;
  for (int a = 0; a < na; ++a) {
    double t = 0.0;
    for (int b = 0; b < nb; ++b) t += ref[a * KS_NREF + b] * y[b];
    acc += x[a] * t;
  }
  return acc;
}

__device__ void d4_pair(const KsClassical& c, const KsWork& w, const KsClsOut& o, const Pair& p,
                        long long pidx) {
  const int i = p.i, j = p.j, ni = it(c, i, KI_NREF), nj = it(c, j, KI_NREF);
  const double* ref = c.c6ref + 49LL * (it(c, i, KI_SP) * c.nsp + it(c, j, KI_SP));
  const double *gi = w.gw + KS_NREF * i, *gj = w.gw + KS_NREF * j;
  // zero-charge C6 and its CN derivatives for the ATM pass
  const double *g0i = w.gw0 + KS_NREF * i, *g0j = w.gw0 + KS_NREF * j;
  w.c60[3 * pidx + 0] = c6_contract(ref, g0i, ni, g0j, nj);
  w.c60[3 * pidx + 1] = c6_contract(ref, w.dgw0dcn + KS_NREF * i, ni, g0j, nj);
  w.c60[3 * pidx + 2] = c6_contract(ref, g0i, ni, w.dgw0dcn + KS_NREF * j, nj);
  if (p.r2 > kDisp2Cut * kDisp2Cut) return;
  double dsw, sw = smooth(p.r, kDisp2Cut, kDispWidth, &dsw);
  if (sw <= 0.0) return;
  double rrij = 3.0 * at(c, i, KA_R4R2) * at(c, j, KA_R4R2);
  double r0 = c.sc[KS_D4_A1] * sqrt(rrij) + c.sc[KS_D4_A2], r02 = r0 * r0;
  double r4 = p.r2 * p.r2, r6 = r4 * p.r2, r8 = r4 * r4;
  double t6 = 1.0 / (r6 + r02 * r02 * r02), t8 = 1.0 / (r8 + r02 * r02 * r02 * r02);
  double d6 = -6.0 * r4 * t6 * t6, d8 = -8.0 * r6 * t8 * t8;  // (dt/dr) / r
  double s6 = c.sc[KS_D4_S6], s8 = c.sc[KS_D4_S8];
  double e0 = s6 * t6 + s8 * rrij * t8, edisp = sw * e0;
  double gdisp = sw * (s6 * d6 + s8 * rrij * d8) + dsw * e0 / p.r;
  double c6 = c6_contract(ref, gi, ni, gj, nj);
  add_energy(w, p, KE_D4TWO, -c6 * edisp);
  double g[3] = {-c6 * gdisp * p.v[0], -c6 * gdisp * p.v[1], -c6 * gdisp * p.v[2]};
  add_force(o, p, g);
  atomicAdd(w.dedcn4 + i, -edisp * c6_contract(ref, w.dgwdcn + KS_NREF * i, ni, gj, nj));
  atomicAdd(w.dedcn4 + j, -edisp * c6_contract(ref, gi, ni, w.dgwdcn + KS_NREF * j, nj));
  atomicAdd(o.vat + i, -edisp * c6_contract(ref, w.dgwdq + KS_NREF * i, ni, gj, nj));
  atomicAdd(o.vat + j, -edisp * c6_contract(ref, gi, ni, w.dgwdq + KS_NREF * j, nj));
}

__device__ void alpb_pair(const KsClassical& c, const KsWork& w, const KsScf& s,
                          const KsClsOut& o, const Pair& p) {
  const int i = p.i, j = p.j;
  double bi = w.brad[i], bj = w.brad[j], ab = sqrt(bi * bj), zr = kZetaP16 / 16.0 * p.r;
  double den = ab + zr, arg = ab / den, a2 = arg * arg, a4 = a2 * a2, a8 = a4 * a4, a16 = a8 * a8;
  double fgb = p.r + ab * a16, keps = c.sc[KS_KEPS];
  double e = keps * s.qat[i] * s.qat[j] / fgb;
  double dfdr = 1.0 - kZetaP16 * arg * a16;
  double dfdab = a16 + 16.0 * ab * (a16 / arg) * zr / (den * den);
  add_energy(w, p, KE_ALPB, e);
  double gs = -e / fgb * dfdr / p.r, g[3] = {gs * p.v[0], gs * p.v[1], gs * p.v[2]};
  add_force(o, p, g);
  double deab = -e / fgb * dfdab;
  atomicAdd(w.dedbr + i, deab * 0.5 * ab / bi);
  atomicAdd(w.dedbr + j, deab * 0.5 * ab / bj);
  atomicAdd(o.vat + i, keps * s.qat[j] / fgb);
  atomicAdd(o.vat + j, keps * s.qat[i] / fgb);
}

__global__ void __launch_bounds__(kThreads) pair_kernel(KsClassical c, KsWork w, KsScf s,
                                                        KsClsOut o) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= c.npair) return;
  Pair p;
  p.i = c.pi[q];
  p.j = c.pj[q];
  p.m = c.atom_mol[p.i];
  p.r2 = pair_vec(c, p.i, p.j, p.v);
  if (p.r2 < 1e-12) return;
  p.r = sqrt(p.r2);
  rep_pair(c, w, o, p);
  es2_pair(c, w, s, o, p);
  aes_pair(c, w, s, o, p);
  d4_pair(c, w, o, p, q);
  if (c.solvent) alpb_pair(c, w, s, o, p);
}

// ------------------------------------------------------------------------------ ATM
// one thread per pair (i > j) within 25 Bohr, looping over k < j; dftd4's
// get_atm_dispersion_derivs with tblite's zero-charge C6 (vij = x_j - x_i there)

__device__ __forceinline__ double atm_dang(double a, double b, double c, double r5) {
  return -0.375 * (a * a * a + a * a * (b + c) + a * (3.0 * b * b + 2.0 * b * c + 3.0 * c * c)
                   - 5.0 * (b - c) * (b - c) * (b + c)) / r5;
}

struct AtmPair {  // the (i, j) leg of a triple, vij = x_j - x_i
  int i, j, m;
  double vij[3], r2ij, rij, swij, dswij, ri, rj, r0ij;
  const double* cij;  // zero-charge C6, dC6/dCN_i, dC6/dCN_j
};

struct AtmAcc {  // what the (i, j) thread accumulates over k
  double gi[3], gj[3], e, dci, dcj;
};

// one triple (i, j, k), k < j < i, added to the accumulators and k's atomics
__device__ void atm_triple(const KsClassical& c, const KsWork& w, const KsClsOut& o,
                           const AtmPair& p, int k, AtmAcc& acc) {
  const double cut2 = kDisp3Cut * kDisp3Cut, eps = 2.220446049250313e-16;
  double vik[3], r2ik = pair_vec(c, k, p.i, vik);
  if (r2ik > cut2 || r2ik < eps) return;
  double vjk[3] = {vik[0] - p.vij[0], vik[1] - p.vij[1], vik[2] - p.vij[2]}, r2jk = dot3(vjk, vjk);
  if (r2jk > cut2 || r2jk < eps) return;
  double rik = sqrt(r2ik), rjk = sqrt(r2jk), dswik, dswjk;
  double swik = smooth(rik, kDisp3Cut, kDispWidth, &dswik);
  double swjk = smooth(rjk, kDisp3Cut, kDispWidth, &dswjk);
  double sw = p.swij * swik * swjk;
  if (sw <= 0.0) return;
  const double *cik = w.c60 + 3 * pair_index(c, p.m, p.i, k);
  const double *cjk = w.c60 + 3 * pair_index(c, p.m, p.j, k);
  const double a1 = c.sc[KS_D4_A1], a2 = c.sc[KS_D4_A2], rk = at(c, k, KA_R4R2);
  double r0 = p.r0ij * (a1 * sqrt(3.0 * p.ri * rk) + a2) * (a1 * sqrt(3.0 * p.rj * rk) + a2);
  double c9 = -c.sc[KS_D4_S9] * sqrt(fabs(p.cij[0] * cik[0] * cjk[0]));
  double r2ij = p.r2ij, r2 = r2ij * r2ik * r2jk, r1 = sqrt(r2), r3 = r2 * r1, r5 = r3 * r2;
  double t = pow(r0 / r1, kAtmAlpha / 3.0), fdmp = 1.0 / (1.0 + 6.0 * t);
  double ang = 0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik) / r5
             + 1.0 / r3;
  double dfdmp = -2.0 * kAtmAlpha * t * fdmp * fdmp, de0 = ang * fdmp * c9;
  double sij = sw * c9 * (-atm_dang(r2ij, r2jk, r2ik, r5) * fdmp + ang * dfdmp) / r2ij
             - de0 * p.dswij / p.rij * swik * swjk;
  double sik = sw * c9 * (-atm_dang(r2ik, r2jk, r2ij, r5) * fdmp + ang * dfdmp) / r2ik
             - de0 * dswik / rik * p.swij * swjk;
  double sjk = sw * c9 * (-atm_dang(r2jk, r2ik, r2ij, r5) * fdmp + ang * dfdmp) / r2jk
             - de0 * dswjk / rjk * p.swij * swik;
  double de = de0 * sw, gk[3];
  for (int x = 0; x < 3; ++x) {
    acc.gi[x] -= sij * p.vij[x] + sik * vik[x];
    acc.gj[x] += sij * p.vij[x] - sjk * vjk[x];
    gk[x] = sik * vik[x] + sjk * vjk[x];
  }
  add3(o.grad, k, gk, 1.0);
  acc.e -= de;
  acc.dci -= de * 0.5 * (p.cij[1] / p.cij[0] + cik[1] / cik[0]);
  acc.dcj -= de * 0.5 * (p.cij[2] / p.cij[0] + cjk[1] / cjk[0]);
  atomicAdd(w.dedcn4 + k, -de * 0.5 * (cik[2] / cik[0] + cjk[2] / cjk[0]));
  atomicAdd(w.eatom + KE_N * k + KE_D4THREE, -de / 3.0);
}

__global__ void __launch_bounds__(kThreads) atm_kernel(KsClassical c, KsWork w, KsClsOut o) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= c.npair) return;
  AtmPair p;
  p.i = c.pi[q];
  p.j = c.pj[q];
  p.m = c.atom_mol[p.i];
  p.r2ij = pair_vec(c, p.j, p.i, p.vij);
  if (p.r2ij > kDisp3Cut * kDisp3Cut || p.r2ij < 2.220446049250313e-16) return;
  p.rij = sqrt(p.r2ij);
  p.swij = smooth(p.rij, kDisp3Cut, kDispWidth, &p.dswij);
  p.ri = at(c, p.i, KA_R4R2);
  p.rj = at(c, p.j, KA_R4R2);
  p.r0ij = c.sc[KS_D4_A1] * sqrt(3.0 * p.ri * p.rj) + c.sc[KS_D4_A2];
  p.cij = w.c60 + 3 * q;
  AtmAcc acc = {{0, 0, 0}, {0, 0, 0}, 0.0, 0.0, 0.0};
  for (int k = c.mol_at0[p.m]; k < p.j; ++k) atm_triple(c, w, o, p, k, acc);
  add3(o.grad, p.i, acc.gi, 1.0);
  add3(o.grad, p.j, acc.gj, 1.0);
  atomicAdd(w.eatom + KE_N * p.i + KE_D4THREE, acc.e / 3.0);
  atomicAdd(w.eatom + KE_N * p.j + KE_D4THREE, acc.e / 3.0);
  atomicAdd(w.dedcn4 + p.i, acc.dci);
  atomicAdd(w.dedcn4 + p.j, acc.dcj);
}

// ------------------------------------------------------------------------------ ALPB shape
// keps alpha / A_det (sum q)^2 / 2, A_det from the inertia of the vdW spheres; one thread per
// molecule. d ln A_det / dx_a = r_a^3 (tr(I^-1) v_a - I^-1 v_a) / 3, the centre term vanishing.

__global__ void adet_kernel(KsClassical c, KsWork w, KsScf s, KsClsOut o) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= c.nmol || !c.solvent) return;
  int a0 = c.mol_at0[m], n = c.mol_nat[m];
  double vol = 0.0, cen[3] = {0, 0, 0}, qsum = 0.0, in[6] = {0, 0, 0, 0, 0, 0};
  for (int a = a0; a < a0 + n; ++a) {
    double r3 = pow(at(c, a, KA_RVDW), 3);
    vol += r3;
    for (int k = 0; k < 3; ++k) cen[k] += r3 * c.xyz[3 * a + k];
    qsum += s.qat[a];
  }
  for (int k = 0; k < 3; ++k) cen[k] /= vol;
  for (int a = a0; a < a0 + n; ++a) {  // inertia xx xy xz yy yz zz
    double rad = at(c, a, KA_RVDW), r3 = rad * rad * rad, v[3];
    for (int k = 0; k < 3; ++k) v[k] = c.xyz[3 * a + k] - cen[k];
    double d = dot3(v, v) + 0.4 * rad * rad;
    in[0] += r3 * (d - v[0] * v[0]);
    in[1] -= r3 * v[0] * v[1];
    in[2] -= r3 * v[0] * v[2];
    in[3] += r3 * (d - v[1] * v[1]);
    in[4] -= r3 * v[1] * v[2];
    in[5] += r3 * (d - v[2] * v[2]);
  }
  double c00 = in[3] * in[5] - in[4] * in[4], c01 = in[2] * in[4] - in[1] * in[5];
  double c02 = in[1] * in[4] - in[2] * in[3], c11 = in[0] * in[5] - in[2] * in[2];
  double c12 = in[1] * in[2] - in[0] * in[4], c22 = in[0] * in[3] - in[1] * in[1];
  double det = in[0] * c00 + in[1] * c01 + in[2] * c02;
  double adet = sqrt(cbrt(det) / (0.4 * vol));
  double kq = c.sc[KS_KEPS] * c.sc[KS_ALPBET] / adet, e = 0.5 * kq * qsum * qsum;
  atomicAdd(w.eatom + KE_N * a0 + KE_ALPB, e);
  double mi[6] = {c00 / det, c01 / det, c02 / det, c11 / det, c12 / det, c22 / det};
  double tr = mi[0] + mi[3] + mi[5];
  for (int a = a0; a < a0 + n; ++a) {
    atomicAdd(o.vat + a, kq * qsum);
    double r3 = pow(at(c, a, KA_RVDW), 3), v[3], g[3];
    for (int k = 0; k < 3; ++k) v[k] = c.xyz[3 * a + k] - cen[k];
    double mv[3] = {mi[0] * v[0] + mi[1] * v[1] + mi[2] * v[2],
                    mi[1] * v[0] + mi[3] * v[1] + mi[4] * v[2],
                    mi[2] * v[0] + mi[4] * v[1] + mi[5] * v[2]};
    for (int k = 0; k < 3; ++k) g[k] = -e / 3.0 * r3 * (tr * v[k] - mv[k]);
    add3(o.grad, a, g, 1.0);
  }
}

// ------------------------------------------------------------------------------ SASA
// one thread per (atom, grid point): the point's smoothed exposure, the atom's surface and,
// weighted by the atom's tension + hbond q^2, the gradient (surface.f90 compute_numsa)

__global__ void __launch_bounds__(kThreads) sasa_kernel(KsClassical c, KsWork w, KsScf s,
                                                        KsClsOut o) {
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (long long)c.nat * KS_NGRID) return;
  const int a = (int)(t / KS_NGRID), pt = (int)(t % KS_NGRID), m = c.atom_mol[a];
  const int a0 = c.mol_at0[m], n = c.mol_nat[m];
  const double wd = c.sc[KS_SASA_W], ah1 = 0.75 / wd, ah3 = -0.25 / (wd * wd * wd);
  double rsa = at(c, a, KA_VDWSA), x[3];
  for (int k = 0; k < 3; ++k) x[k] = c.xyz[3 * a + k] + rsa * c.grid[3 * pt + k];
  double sasap = 1.0;
  for (int b = a0; b < a0 + n && sasap > 0.0; ++b) {
    if (b == a) continue;
    double tb[3], rb = at(c, b, KA_VDWSA);
    for (int k = 0; k < 3; ++k) tb[k] = x[k] - c.xyz[3 * b + k];
    double t2 = dot3(tb, tb);
    if (t2 >= (rb + wd) * (rb + wd)) continue;
    if (t2 <= (rb - wd) * (rb - wd)) {
      sasap = 0.0;
      break;
    }
    double u = sqrt(t2) - rb;
    sasap *= 0.5 + (ah1 + ah3 * u * u) * u;
  }
  if (!(sasap > kTolSesp)) return;
  double wsa = c.gweight[pt] * at(c, a, KA_WRP) * sasap;
  atomicAdd(w.surface + a, wsa);
  double q = s.qat[a], wt = at(c, a, KA_TENSION) + at(c, a, KA_HBOND) * q * q;
  double ga[3] = {0, 0, 0};
  for (int b = a0; b < a0 + n; ++b) {
    if (b == a) continue;
    double tb[3], rb = at(c, b, KA_VDWSA);
    for (int k = 0; k < 3; ++k) tb[k] = x[k] - c.xyz[3 * b + k];
    double t2 = dot3(tb, tb);
    if (t2 >= (rb + wd) * (rb + wd)) continue;
    double st = sqrt(t2), u = st - rb, f = 0.5 + (ah1 + ah3 * u * u) * u;
    double sc = wt * wsa * (ah1 + 3.0 * ah3 * u * u) / (f * st);
    for (int k = 0; k < 3; ++k) ga[k] += sc * tb[k];
    add3(o.grad, b, tb, -sc);
  }
  add3(o.grad, a, ga, 1.0);
}

// ------------------------------------------------------------------------------ post / chain

__global__ void __launch_bounds__(kThreads) post_kernel(KsClassical c, KsWork w, KsScf s,
                                                        KsClsOut o, const double* dedcn_in) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= c.nat) return;
  w.dedcn[a] = w.dedcn[a] * w.dmrdcn[a] + (dedcn_in ? dedcn_in[a] : 0.0);
  if (!c.solvent) return;
  w.dedbr[a] *= w.dbrdpsi[a];  // now dE/dpsi
  double q = s.qat[a], hb = at(c, a, KA_HBOND), area = w.surface[a];
  w.eatom[KE_N * a + KE_CDS] += area * (at(c, a, KA_TENSION) + hb * q * q);
  o.vat[a] += 2.0 * hb * area * q;
}

__global__ void __launch_bounds__(kThreads) chain_kernel(KsClassical c, KsWork w, KsClsOut o) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= c.npair) return;
  int i = c.pi[q], j = c.pj[q];
  double v[3], r2 = pair_vec(c, i, j, v);
  if (r2 < 1e-12) return;
  double r = sqrt(r2), f, df, s = 0.0;
  if (r2 <= kCnDexpCut2) {
    dexp_count(c, i, j, r, &f, &df);
    s += (w.dedcn[i] + w.dedcn[j]) * df / r;
  }
  if (r2 <= kCnD4Cut2) {
    d4_count(c, i, j, r, &f, &df);
    s += (w.dedcn4[i] + w.dedcn4[j]) * df / r;
  }
  if (c.solvent && r2 <= kBornCut2) {
    double dgi, dgj;
    born_g(r, at(c, j, KA_RHO), at(c, i, KA_RVDW), &dgi);
    born_g(r, at(c, i, KA_RHO), at(c, j, KA_RVDW), &dgj);
    s += w.dedbr[i] * dgi + w.dedbr[j] * dgj;
  }
  add3(o.grad, i, v, s);
  add3(o.grad, j, v, -s);
}

__global__ void reduce_kernel(KsClassical c, KsWork w, KsClsOut o) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= c.nmol * KE_N) return;
  int m = t / KE_N, term = t % KE_N, a0 = c.mol_at0[m];
  double e = 0.0;
  for (int a = a0; a < a0 + c.mol_nat[m]; ++a) e += w.eatom[KE_N * a + term];
  o.energy[t] = e;
}

inline unsigned blocks(long long n) { return (unsigned)((n + kThreads - 1) / kThreads); }

inline cudaError_t zero(void* p, size_t bytes, cudaStream_t st) {
  return bytes ? cudaMemsetAsync(p, 0, bytes, st) : cudaSuccess;
}

}  // namespace

extern "C" int ksxtb_geometry(const KsClassical* cp, const KsWork* wp, void* stream) {
  const KsClassical c = *cp;
  const KsWork w = *wp;
  cudaStream_t st = (cudaStream_t)stream;
  const size_t na = sizeof(double) * c.nat;
  cudaError_t e;
  if ((e = zero(w.cn, na, st)) || (e = zero(w.cn4, na, st)) || (e = zero(w.psi, na, st)))
    return (int)e;
  if (c.npair) cn_psi_kernel<<<blocks(c.npair), kThreads, 0, st>>>(c, w);
  if (c.nat) atom_geometry_kernel<<<blocks(c.nat), kThreads, 0, st>>>(c, w);
  return (int)cudaGetLastError();
}

extern "C" int ksxtb_classical(const KsClassical* cp, const KsWork* wp, const KsScf* sp,
                               const KsClsOut* op, const double* dedcn_in, void* stream) {
  const KsClassical c = *cp;
  const KsWork w = *wp;
  const KsScf s = *sp;
  const KsClsOut o = *op;
  cudaStream_t st = (cudaStream_t)stream;
  const size_t na = sizeof(double) * c.nat;
  cudaError_t e;
  if ((e = zero(w.dedcn, na, st)) || (e = zero(w.dedcn4, na, st)) || (e = zero(w.dedbr, na, st)) ||
      (e = zero(w.surface, na, st)) || (e = zero(w.eatom, KE_N * na, st)) ||
      (e = zero(o.vsh, sizeof(double) * c.nsh, st)) || (e = zero(o.vat, na, st)) ||
      (e = zero(o.vdp, 3 * na, st)) || (e = zero(o.vqp, 6 * na, st)) ||
      (e = zero(o.grad, 3 * na, st)))
    return (int)e;
  if (c.nat == 0) return (int)cudaSuccess;
  atom_scf_kernel<<<blocks(c.nat), kThreads, 0, st>>>(c, w, s, o);
  if (c.npair) {
    pair_kernel<<<blocks(c.npair), kThreads, 0, st>>>(c, w, s, o);
    atm_kernel<<<blocks(c.npair), kThreads, 0, st>>>(c, w, o);
  }
  if (c.solvent) {
    adet_kernel<<<blocks(c.nmol), kThreads, 0, st>>>(c, w, s, o);
    sasa_kernel<<<blocks((long long)c.nat * KS_NGRID), kThreads, 0, st>>>(c, w, s, o);
  }
  post_kernel<<<blocks(c.nat), kThreads, 0, st>>>(c, w, s, o, dedcn_in);
  if (c.npair) chain_kernel<<<blocks(c.npair), kThreads, 0, st>>>(c, w, o);
  reduce_kernel<<<blocks((long long)c.nmol * KE_N), kThreads, 0, st>>>(c, w, o);
  return (int)cudaGetLastError();
}
