// SyntheticBanner: warns that the numbers do not come from GFN2-xTB (design §2.6, §11.1).
//
// Props:
//   meta   MetaOut | null    from GET /api/meta (state/store.ts metaStore); nothing renders
//                            until it is known, or when chemistry is "real" or "cached"
//
// DOM: <div data-testid="synthetic-banner" data-chemistry="synthetic|stub">.
import type { MetaOut } from "../api/types.gen";

export interface SyntheticBannerProps {
  meta: MetaOut | null | undefined;
}

export function SyntheticBanner({ meta }: SyntheticBannerProps) {
  if (!meta || (meta.chemistry !== "synthetic" && meta.chemistry !== "stub")) return null;
  const what =
    meta.chemistry === "synthetic"
      ? `synthetic chemistry (${meta.qm_method}): charge-iterated extended Hückel on the GFN2 basis, not GFN2-xTB`
      : "stub chemistry: table-driven test species, no quantum chemistry";
  return (
    <div className="banner banner-warn" data-testid="synthetic-banner" data-chemistry={meta.chemistry} role="status">
      <strong>Synthetic data.</strong> {what}. Scores, orbitals and physics are for UI development and tests
      only; do not curate from them.
    </div>
  );
}
