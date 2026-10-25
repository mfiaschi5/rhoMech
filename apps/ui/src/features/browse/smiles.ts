// Unmapped species of a step side, for display and for the "predict from these reactants" link
// (#/predict?r=...&n=..., design §10, §11.8). Atom maps are stripped from the bracket atoms (in
// SMILES ":<digits>]" occurs only as an atom-map class), then RDKit.js canonicalizes each fragment
// and folds explicit H atoms into their parents. Without RDKit.js the stripped fragment is used
// as is (still a valid SMILES). Repeated species are counted, so 2 x water gives n=2.
import type { RDKitModule } from "../../chem/rdkit";
import { withMol } from "../../chem/rdkit";
import type { PrefillSpecies } from "../../router";

export function stripMaps(smiles: string): string {
  return smiles.replace(/:\d+\]/g, "]");
}

export function canonicalFragment(rdkit: RDKitModule | null, fragment: string): string {
  const stripped = stripMaps(fragment.trim());
  if (!rdkit || !stripped) return stripped;
  const canon = withMol(rdkit, stripped, { removeHs: true }, (mol) => mol.get_smiles());
  return canon || stripped;
}

/** The species of a side, unmapped and counted, in first-seen order. */
export function speciesCounts(rdkit: RDKitModule | null, sideSmiles: string): PrefillSpecies[] {
  const counts = new Map<string, number>();
  for (const frag of sideSmiles.split(".")) {
    if (!frag.trim()) continue;
    const s = canonicalFragment(rdkit, frag);
    counts.set(s, (counts.get(s) ?? 0) + 1);
  }
  return [...counts].map(([smiles, count]) => ({ smiles, count }));
}

/** "2 × O + CCl" */
export function speciesText(species: readonly PrefillSpecies[]): string {
  return species.map((s) => `${s.count > 1 ? `${s.count} × ` : ""}${s.smiles}`).join(" + ");
}
