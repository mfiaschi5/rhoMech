// Number and label formatting shared by the pages (Predict, Correct, Workbench).
//
// Values that carry a data-testid (physics-ks3, floor-rank, ...) are written with an ASCII
// "-" so a test can parse them with Number(); typographic minus signs appear only in prose
// labels (HOMO−1, "−P +S").
import type { CentreOut, OrbitalDescBrief } from "../api/types.gen";

/** A fixed-decimal number, or "–" for null / NaN / inf. */
export function fmt(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  return value.toFixed(digits);
}

/** A number with `sig` significant digits (exponent form for very small or large values). */
export function fmtSig(value: number | null | undefined, sig = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  if (value === 0) return "0";
  const mag = Math.abs(value);
  if (mag < 1e-3 || mag >= 1e5) return value.toExponential(sig - 1);
  return Number(value.toPrecision(sig)).toString();
}

/** A probability in [0, 1] with three decimals ("–" when unknown). */
export function fmtProb(p: number | null | undefined): string {
  return fmt(p, 3);
}

/** The full-precision value for hover titles and data-value attributes. */
export function exact(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "";
  return String(value);
}

/** "HOMO-1" from the server becomes "HOMO−1" for display. */
export function moLabel(label: string): string {
  return label.replace(/([A-Z])-(\d)/g, "$1−$2");
}

/** "j1a2b3c4…" for long ids in prose. */
export function shortId(id: string | null | undefined, n = 9): string {
  if (!id) return "–";
  return id.length > n ? `${id.slice(0, n)}…` : id;
}

/** "O1 0.98" for one centre (element, map number, centre weight). */
export function centreText(c: CentreOut): string {
  return `${c.element}${c.map} ${fmt(c.weight, 2)}`;
}

/** The model-input descriptors of an orbital token as one line of plain text. */
export function descText(d: OrbitalDescBrief): string {
  const centres = d.centres.map(centreText).join(", ");
  const spd = d.spd0.map((v) => fmt(v, 2)).join("/");
  const spectral = d.spectral_set
    .slice(0, 3)
    .map(([e, w]) => `${fmt(e, 2)} (${fmt(w, 2)})`)
    .join(", ");
  const more = d.spectral_set.length > 3 ? ` +${d.spectral_set.length - 3}` : "";
  return (
    `occ ${fmt(d.occupation, 2)} · ${d.n_centers} centre${d.n_centers === 1 ? "" : "s"}: ${centres}` +
    ` · s/p/d ${spd} · |cos| ${fmt(d.cos_a, 2)}/${fmt(d.cos_b, 2)} · overlap sign ${d.overlap_sign}` +
    ` · spectral ${spectral}${more}`
  );
}

/** Seconds as "12 s" / "3 min 4 s". */
export function fmtSeconds(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return "–";
  if (s < 60) return `${Math.round(s)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - 60 * m)} s`;
}

/** Nice axis ticks covering [lo, hi] (at most about `n` of them). */
export function niceTicks(lo: number, hi: number, n = 4): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [lo];
  const raw = (hi - lo) / Math.max(1, n);
  const pow = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * pow).find((s) => s >= raw) ?? 10 * pow;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : Number(v.toPrecision(12)));
  }
  return out;
}

/** A short tick label. */
export function tickText(v: number): string {
  if (v === 0) return "0";
  const mag = Math.abs(v);
  if (mag < 1e-3 || mag >= 1e4) return v.toExponential(1);
  return Number(v.toPrecision(3)).toString();
}
