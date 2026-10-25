// The StepDoc of a curated step (design §9.2, format "ksmech-serve/step-v1"), as CuratedStepOut
// carries it: an untyped dict. parseStepDoc reads it defensively; an arrow that lacks the fields
// needed to draw it is not dropped silently but counted in `badArrows` and shown as a warning.
import type { DatasetArrowOut } from "../../api/types.gen";
import type { TokenArrowLike } from "../../chem/arrowGeom";

export const STEP_DOC_FORMAT = "ksmech-serve/step-v1";

export interface StepArrow extends TokenArrowLike {
  source_kind: string;
}

export interface StepDocView {
  format: string;
  jobId: string | null;
  edgeId: string | null;
  edgeKind: string | null;
  reactantStateId: string | null;
  productStateId: string | null;
  reactantsMapped: string;
  productsMapped: string;
  arrows: StepArrow[];
  badArrows: number;
  datasetArrows: DatasetArrowOut[] | null;
  pmechdbCode: string | null;
  encodeReason: string | null;
  score: number | null;
  probability: number | null;
  rank: number | null;
  predictor: string | null;
  predictorVersion: string | null;
  specVersion: string | null;
  physicsFingerprint: string | null;
  touchedMaps: number[];
}

type Dict = Record<string, unknown>;

const str = (d: Dict, k: string): string | null => (typeof d[k] === "string" ? (d[k] as string) : null);
const num = (d: Dict, k: string): number | null =>
  typeof d[k] === "number" && Number.isFinite(d[k]) ? (d[k] as number) : null;
const ints = (v: unknown): number[] =>
  Array.isArray(v) ? v.filter((x): x is number => typeof x === "number" && Number.isInteger(x)) : [];
const isDict = (v: unknown): v is Dict => !!v && typeof v === "object" && !Array.isArray(v);

function parseArrow(raw: unknown): StepArrow | null {
  if (!isDict(raw)) return null;
  const need = ["donor", "donor_site", "acceptor", "acceptor_site", "donor_site_map", "acceptor_site_map"];
  const vals = need.map((k) => num(raw, k));
  if (vals.some((v) => v === null)) return null;
  const [donor, donorSite, acceptor, acceptorSite, donorMap, acceptorMap] = vals as number[];
  return {
    donor,
    donor_site: donorSite,
    acceptor,
    acceptor_site: acceptorSite,
    donor_site_map: donorMap,
    acceptor_site_map: acceptorMap,
    donor_element: str(raw, "donor_element") ?? "?",
    acceptor_element: str(raw, "acceptor_element") ?? "?",
    source_partner_map: num(raw, "source_partner_map"),
    sink_partner_map: num(raw, "sink_partner_map"),
    source_kind: str(raw, "source_kind") ?? "unknown",
    electrons: num(raw, "electrons") === 1 ? 1 : 2,
  };
}

function parseDatasetArrows(raw: unknown): DatasetArrowOut[] | null {
  if (!Array.isArray(raw)) return null;
  return raw.filter(isDict).map((a) => ({
    source_kind: str(a, "source_kind") ?? "unknown",
    source_atoms: ints(a.source_atoms),
    sink_kind: str(a, "sink_kind") ?? "unknown",
    sink_atoms: ints(a.sink_atoms),
  }));
}

export function parseStepDoc(raw: unknown): StepDocView {
  const d: Dict = isDict(raw) ? raw : {};
  const rawArrows = Array.isArray(d.arrows) ? d.arrows : [];
  const arrows = rawArrows.map(parseArrow).filter((a): a is StepArrow => a !== null);
  return {
    format: str(d, "format") ?? "",
    jobId: str(d, "job_id"),
    edgeId: str(d, "edge_id"),
    edgeKind: str(d, "edge_kind"),
    reactantStateId: str(d, "reactant_state_id"),
    productStateId: str(d, "product_state_id"),
    reactantsMapped: str(d, "reactants_mapped") ?? "",
    productsMapped: str(d, "products_mapped") ?? "",
    arrows,
    badArrows: rawArrows.length - arrows.length,
    datasetArrows: parseDatasetArrows(d.dataset_arrows),
    pmechdbCode: str(d, "pmechdb_code"),
    encodeReason: str(d, "encode_reason"),
    score: num(d, "score"),
    probability: num(d, "probability"),
    rank: num(d, "rank"),
    predictor: str(d, "predictor"),
    predictorVersion: str(d, "predictor_version"),
    specVersion: str(d, "spec_version"),
    physicsFingerprint: str(d, "physics_fingerprint"),
    touchedMaps: ints(d.touched_maps),
  };
}

/** Every map number an arrow list draws (the H atoms among them must stay in the depiction). */
export function arrowMaps(
  arrows: readonly TokenArrowLike[] | null | undefined,
  dataset?: readonly DatasetArrowOut[] | null,
): number[] {
  const maps = new Set<number>();
  for (const a of arrows ?? []) {
    for (const m of [a.donor_site_map, a.acceptor_site_map, a.source_partner_map, a.sink_partner_map]) {
      if (typeof m === "number") maps.add(m);
    }
  }
  for (const a of dataset ?? []) for (const m of [...a.source_atoms, ...a.sink_atoms]) maps.add(m);
  return [...maps].sort((x, y) => x - y);
}

export function fmtProb(p: number | null | undefined): string {
  return p === null || p === undefined ? "–" : p.toFixed(3);
}

export function fmtNum(v: number | null | undefined, digits = 3): string {
  return v === null || v === undefined || !Number.isFinite(v) ? "–" : v.toFixed(digits);
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "–";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/** A token arrow as text: "O(2) → C(6)", with the bond source and implied sink when known. */
export function arrowText(a: TokenArrowLike): string {
  const src = a.source_partner_map != null ? `${a.donor_element}${a.donor_site_map}–${a.source_partner_map}` : `${a.donor_element}${a.donor_site_map}`;
  const sink = a.sink_partner_map != null ? ` (breaks ${a.acceptor_site_map}–${a.sink_partner_map})` : "";
  return `${src} → ${a.acceptor_element}${a.acceptor_site_map}${sink}`;
}

export function datasetArrowText(a: DatasetArrowOut): string {
  return `${a.source_kind} [${a.source_atoms.join(",")}] → ${a.sink_kind} [${a.sink_atoms.join(",")}]`;
}
