/* ksxtb: batched GFN2-xTB building blocks on CUDA (stage 1: one-electron integrals, stage 2:
 * the remaining terms with their potentials and gradients).
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

/* ---- stage 2: the GFN2 terms beyond the one-electron integrals (classical.cu) ----------------
 * Repulsion, ES2, ES3, AES, D4 (two-body with the SCF charges, ATM with zero charges), ALPB
 * and CDS of a batch, as `ksmech.qm.gpu.terms` writes them. Per-atom parameters sit in two
 * row-major tables, atab[nat * KA_COLS] and itab[nat * KI_COLS]. The pair list holds every
 * pair i > j of each molecule in the order (i_loc = 1..n-1, j_loc = 0..i_loc-1) from
 * mol_pair0[m], so the pair index of (i, j) is mol_pair0[m] + i_loc (i_loc - 1) / 2 + j_loc. */

enum { KS_NREF = 7, KS_NGRID = 230 };

enum { /* atab columns */
  KA_RCOV, KA_RCOV4, KA_EN4, KA_REP_ALPHA, KA_REP_ZEFF, KA_MP_RAD, KA_MP_VCN, KA_DKERNEL,
  KA_QKERNEL, KA_R4R2, KA_D4_ZI, KA_D4_GI, KA_RVDW, KA_RHO, KA_SVDW, KA_VDWSA, KA_WRP,
  KA_TENSION, KA_HBOND, KA_REFCN, KA_REFQ = KA_REFCN + KS_NREF, KA_COLS = KA_REFQ + KS_NREF
};
enum { /* itab columns: atomic number, species of the batch, D4 references and their
          Gaussian counts, first shell and shell count */
  KI_Z, KI_SP, KI_NREF, KI_NGW, KI_SH0 = KI_NGW + KS_NREF, KI_NSH, KI_COLS
};
enum { /* scalars, KsClassical.sc */
  KS_REP_KEXP, KS_REP_KEXP_LIGHT, KS_REP_REXP, KS_MP_DMP3, KS_MP_DMP5, KS_MP_SHIFT,
  KS_MP_KEXP, KS_MP_RMAX, KS_D4_S6, KS_D4_S8, KS_D4_A1, KS_D4_A2, KS_D4_S9, KS_D4_GA,
  KS_D4_WF, KS_KEPS, KS_ALPBET, KS_BORN_C1, KS_SASA_W, KS_NSC
};
enum { /* energy terms, per molecule */
  KE_REP, KE_ES2, KE_ES3, KE_AES, KE_D4TWO, KE_D4THREE, KE_ALPB, KE_CDS, KE_N
};

typedef struct {
  const double* xyz;      /* [nat * 3] */
  const int* atom_mol;    /* [nat] */
  const double* atab;     /* [nat * KA_COLS] */
  const int* itab;        /* [nat * KI_COLS] */
  const int* sh_atom;     /* [nsh] */
  const double* sh_eta;   /* [nsh] ES2 shell hardness */
  const double* sh_hd;    /* [nsh] ES3 shell Hubbard derivative */
  const int* pi;          /* [npair] i > j */
  const int* pj;          /* [npair] */
  const int* mol_at0;     /* [nmol] first atom */
  const int* mol_nat;     /* [nmol] */
  const long long* mol_pair0; /* [nmol] first pair */
  const double* c6ref;    /* [nsp * nsp * 49] reference C6 by species pair, [a * 7 + b] */
  const double* grid;     /* [KS_NGRID * 3] Lebedev points */
  const double* gweight;  /* [KS_NGRID] weights, summing to 4 pi */
  double sc[KS_NSC];
  int nat, nsh, nmol, npair, nsp, solvent;
} KsClassical;

typedef struct { /* device workspace, every array sized by atoms, pairs or molecules */
  double* cn;      /* [nat] dexp CN (H0 self-energies, multipole radii) */
  double* cn4;     /* [nat] D4 CN */
  double* mrad;    /* [nat] multipole radii */
  double* dmrdcn;  /* [nat] */
  double* psi;     /* [nat] Born descreening sum */
  double* brad;    /* [nat] Born radii */
  double* dbrdpsi; /* [nat] */
  double* gw;      /* [nat * 7] D4 weights at the SCF charges */
  double* dgwdcn;  /* [nat * 7] */
  double* dgwdq;   /* [nat * 7] */
  double* gw0;     /* [nat * 7] D4 weights at zero charge (ATM) */
  double* dgw0dcn; /* [nat * 7] */
  double* c60;     /* [npair * 3] zero-charge C6, dC6/dCN_i, dC6/dCN_j (ATM) */
  double* dedcn;   /* [nat] dE/dCN, dexp CN */
  double* dedcn4;  /* [nat] dE/dCN, D4 CN */
  double* dedbr;   /* [nat] dE/dBorn radius */
  double* surface; /* [nat] SASA */
  double* eatom;   /* [nat * KE_N] */
} KsWork;

typedef struct { /* SCF variables */
  const double* qsh; /* [nsh] */
  const double* qat; /* [nat] */
  const double* dp;  /* [nat * 3] */
  const double* qp;  /* [nat * 6] xx xy yy xz yz zz */
} KsScf;

typedef struct { /* outputs, overwritten */
  double* energy; /* [nmol * KE_N] */
  double* vsh;    /* [nsh] dE/dqsh (ES2, ES3) */
  double* vat;    /* [nat] dE/dqat (AES, D4, ALPB, CDS) */
  double* vdp;    /* [nat * 3] */
  double* vqp;    /* [nat * 6] */
  double* grad;   /* [nat * 3] dE/dR at fixed SCF variables, CN, Born and surface chains included */
} KsClsOut;

/* Coordination numbers, multipole radii and Born radii of the current geometry. */
int ksxtb_geometry(const KsClassical* c, const KsWork* w, void* stream);

/* Energies, potentials and gradient at the SCF variables. `dedcn_in` (may be NULL) is an
 * external dE/dCN of the dexp CN, the H0 self-energies' from ksxtb_h0_gradient, chained into
 * the gradient with the multipole radii's. Needs ksxtb_geometry of the same geometry. */
int ksxtb_classical(const KsClassical* c, const KsWork* w, const KsScf* s, const KsClsOut* o,
                    const double* dedcn_in, void* stream);

#ifdef __cplusplus
}
#endif
#endif
