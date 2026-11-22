/* ksxtb: batched GFN2-xTB building blocks on CUDA (stage 1: one-electron integrals).
 *
 * All arrays are device pointers. A batch packs many molecules: atoms, shells and AO matrices
 * are concatenated, each molecule's nao x nao matrices stored row-major at `mol_mat[m]`
 * (multi-component matrices at `ncomp * mol_mat[m]`, component-major). Conventions follow
 * tblite 0.7.0 and `ksmech.qm.gpu.reference`. Every entry point returns a cudaError_t code. */
#ifndef KSXTB_H
#define KSXTB_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const double* xyz;        /* [nat * 3] Bohr */
  const double* atom_rad;   /* [nat] H0 atomic radius, Bohr */
  const int* atom_mol;      /* [nat] molecule of each atom */
  const int* sh_atom;       /* [nsh] atom of each shell */
  const int* sh_cg;         /* [nsh] row of the contracted-Gaussian table */
  const int* sh_ao;         /* [nsh] first AO of the shell within its molecule */
  const double* sh_shpoly;  /* [nsh] H0 shell polynomial */
  const double* sh_se;      /* [nsh] self-energy, Hartree */
  const double* sh_dsedcn;  /* [nsh] d(self-energy)/dCN */
  const int* cg_nprim;      /* [ncg] */
  const double* cg_alpha;   /* [ncg * 6] */
  const double* cg_coeff;   /* [ncg * 6] normalized contraction coefficients */
  const long long* mol_mat; /* [nmol] offset of the molecule's nao x nao block */
  const int* mol_nao;       /* [nmol] */
  const int* mol_ao;        /* [nmol] offset of the molecule's AOs in AO-indexed vectors */
  const double* mol_cut2;   /* [nmol] squared neighbour cutoff, Bohr^2 */
  int nat, nsh, nmol;
} KsBatch;

/* Shell pairs of one angular class. Off-site pairs: si on atom A, sj on atom B, A > B (one
 * entry per unordered pair). On-site pairs: every ordered pair of shells of one atom. */
typedef struct {
  const int* si;
  const int* sj;
  const double* hscale; /* [n] H0 scale of the pair (unused on-site) */
  int n, li, lj, onsite;
} KsPairs;

typedef struct {
  double* S;  /* [mat] */
  double* H0; /* [mat] */
  double* D;  /* [3 * mat] D[c, mu, nu] = <mu|(r - R_atom(nu))_c|nu> */
  double* Q;  /* [6 * mat] traceless, xx xy yy xz yz zz, centred like D */
} KsFwdOut;

typedef struct {
  const double* P;   /* [mat] density */
  const double* X;   /* [mat] energy-weighted density */
  const double* vao; /* [nao_total] AO potential, at mol_ao[m] + mu */
  const double* vdp; /* [nat * 3] atomic dipole potential */
  const double* vqp; /* [nat * 6] atomic quadrupole potential */
} KsGradIn;

typedef struct {
  double* grad;  /* [nat * 3], accumulated (zero it first) */
  double* dedcn; /* [nat], accumulated */
} KsGradOut;

/* S, H0, D and Q of every listed pair, both triangles. The pair lists must cover every AO
 * pair of every molecule (all off-site pairs and all on-site pairs), so no memset is needed. */
int ksxtb_forward(const KsBatch* b, const KsPairs* lists, int nlists, const KsFwdOut* out,
                  void* stream);

/* tblite's get_hamiltonian_gradient for closed shells: off-site lists only, plus the on-site
 * dE/dCN term of the self-energies. */
int ksxtb_h0_gradient(const KsBatch* b, const KsPairs* lists, int nlists, const KsGradIn* in,
                      const KsGradOut* out, void* stream);

#ifdef __cplusplus
}
#endif
#endif
