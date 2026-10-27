// Element data of the sketcher: the palette, the periodic picker layout, the default valences
// used for implicit H counts until RDKit has judged the drawing, and the Hill formula. Pure.

export const PALETTE = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"] as const;

/** Periods 1-6 of the periodic table, 18 columns ("" = gap). The lanthanides are left out
 *  (La stands in group 3); Fr onwards is not offered. */
export const PERIODIC_ROWS: readonly (readonly string[])[] = [
  ["H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"],
  ["Li", "Be", "", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"],
  ["Na", "Mg", "", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"],
  ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
  ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"],
  ["Cs", "Ba", "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
];

const ALL = new Set(PERIODIC_ROWS.flat().filter(Boolean));

export function isElement(symbol: string): boolean {
  return ALL.has(symbol);
}

/** Neutral default valence of the organic subset (others take no implicit H). */
const VALENCE: Record<string, number> = {
  H: 1, B: 3, C: 4, N: 3, O: 2, F: 1, Si: 4, P: 3, S: 2, Cl: 1, Ge: 4, As: 3, Se: 2, Br: 1, Te: 2, I: 1,
};
const GROUP_13_14 = new Set(["B", "C", "Si", "Ge"]);

/** Implicit H count from default valences, the stand-in until RDKit's count arrives: groups 13
 *  and 14 lose one valence per charge of either sign, groups 15-17 gain one per positive charge
 *  (N+ 4, O+ 3) and lose one per negative charge (N- 2, O- 1); each unpaired electron takes one
 *  more (CH3-CH2. has 2 H, :CH2 has 2). */
export function fallbackHCount(el: string, charge: number, bondOrderSum: number, radical = 0): number {
  const v = VALENCE[el];
  if (v === undefined) return 0;
  const valence = GROUP_13_14.has(el) ? v - Math.abs(charge) : v + charge;
  return Math.max(0, valence - bondOrderSum - radical);
}

/** Hill formula (C, H, then alphabetical; alphabetical throughout without carbon) with the net
 *  charge as a suffix: "C2H6O", "C2H8N+", "C2H3O2-". */
export function hillFormula(counts: ReadonlyMap<string, number>, charge = 0): string {
  const els = [...counts.keys()].filter((e) => (counts.get(e) ?? 0) > 0);
  const hasC = els.includes("C");
  const order = hasC
    ? ["C", ...(els.includes("H") ? ["H"] : []), ...els.filter((e) => e !== "C" && e !== "H").sort()]
    : els.sort();
  const body = order.map((e) => (counts.get(e) === 1 ? e : `${e}${counts.get(e)}`)).join("");
  const q = charge === 0 ? "" : `${Math.abs(charge) === 1 ? "" : Math.abs(charge)}${charge > 0 ? "+" : "-"}`;
  return body + q;
}

/** The CSS class of an element label (colour family). */
export function elementClass(el: string): string {
  if (["N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "B", "Si"].includes(el)) return `el-${el}`;
  return el === "C" ? "el-C" : "el-other";
}

/** A charge as a label suffix: "+", "2+", "-" (with a true minus sign), "". */
export function chargeLabel(charge: number): string {
  if (charge === 0) return "";
  const n = Math.abs(charge);
  return `${n === 1 ? "" : n}${charge > 0 ? "+" : "−"}`;
}
