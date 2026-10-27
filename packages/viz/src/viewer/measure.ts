// Distances, angles and dihedrals between record atoms (Angstrom, degrees), for the measure
// tool of the 3D view. Pure functions.
export type MeasureMode = "select" | "distance" | "angle" | "dihedral";

export const MEASURE_ATOMS: Record<Exclude<MeasureMode, "select">, number> = { distance: 2, angle: 3, dihedral: 4 };

export type P3 = readonly [number, number, number];

export interface Measurement {
  mode: Exclude<MeasureMode, "select">;
  atoms: number[];
  value: number;
}

const sub = (a: P3, b: P3): P3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: P3, b: P3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: P3, b: P3): P3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: P3) => Math.sqrt(dot(a, a));
const deg = (r: number) => (r * 180) / Math.PI;

export function distance(a: P3, b: P3): number {
  return norm(sub(a, b));
}

/** The angle a-b-c at b. */
export function angle(a: P3, b: P3, c: P3): number {
  const u = sub(a, b);
  const v = sub(c, b);
  const cos = dot(u, v) / (norm(u) * norm(v) || 1);
  return deg(Math.acos(Math.max(-1, Math.min(1, cos))));
}

/** The signed dihedral a-b-c-d in (-180, 180]. */
export function dihedral(a: P3, b: P3, c: P3, d: P3): number {
  const b1 = sub(b, a);
  const b2 = sub(c, b);
  const b3 = sub(d, c);
  const n1 = cross(b1, b2);
  const n2 = cross(b2, b3);
  const m1 = cross(n1, b2);
  const x = dot(n1, n2);
  const y = dot(m1, n2) / (norm(b2) || 1);
  return deg(Math.atan2(-y, x));
}

export function measure(mode: Exclude<MeasureMode, "select">, pts: readonly P3[]): number {
  if (mode === "distance") return distance(pts[0], pts[1]);
  if (mode === "angle") return angle(pts[0], pts[1], pts[2]);
  return dihedral(pts[0], pts[1], pts[2], pts[3]);
}

export function measureText(m: Measurement): string {
  return m.mode === "distance" ? `${m.value.toFixed(3)} Å` : `${m.value.toFixed(1)}°`;
}
