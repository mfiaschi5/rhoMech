// The spec-v9 serve contract the Predict and Correct pages are written against
// (docs/serve_v9_contract.md), in one place: re-exports of the generated api/types.gen.ts under
// the names the pages use. Atom indices are pool atoms (parse_mapped(state.mapped_smiles),
// explicit H included): DepictAtomOut.pool_atom, never depiction_idx or global_site.
export type {
  AnchorOut,
  ArrowReadingOut,
  CanonicalPairOut,
  CycleOut,
  EdgeOut,
  ExpandIn,
  ExpandResultOut,
  MatrixIO,
  NetParamsOut,
  NetworkOut,
  NodeOut,
  OrbitalReadingOut,
  OverrideIn as OverrideV9In,
  OverrideResultOut as OverrideV9Out,
  PathwayOut,
  PredictIn,
  PredictorInfoOut as PredictorOut,
  ProgressOut,
  Provenance,
  StepCheckIn,
  StepCheckOut,
  StepIssueOut,
} from "../../api/types.gen";

export type ArrowKind = "lone_pair_to_bond" | "bond_to_bond" | "bond_to_lone_pair";
