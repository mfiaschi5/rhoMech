// Gaussian cube text for scene assets (scripts/scene-export.ts): a coarser copy of a cube that
// keeps every n-th grid point along each axis (the axis vectors grow n times, the origin, atoms
// and values stay as the server wrote them). The workbench serves its cubes at a fixed fine
// spacing; a stride of 2 makes a scene asset about 8 times smaller.

interface Cube {
  comments: [string, string];
  natoms: string;
  dims: [number, number, number];
  axes: number[][];
  atoms: string[];
  /** The orbital-index line of a cube with a negative atom count. */
  extra: string[];
  values: string[];
}

function parseCube(text: string): Cube {
  const lines = text.split(/\r?\n/);
  const natoms = lines[2].trim().split(/\s+/);
  const n = Math.abs(Number.parseInt(natoms[0], 10));
  const axes = lines.slice(3, 6).map((l) => l.trim().split(/\s+/).map(Number));
  const dims = axes.map((a) => a[0]) as [number, number, number];
  const extra = Number.parseInt(natoms[0], 10) < 0 ? [lines[6 + n]] : [];
  const values = lines.slice(6 + n + extra.length).join(" ").trim().split(/\s+/).filter(Boolean);
  if (values.length !== dims[0] * dims[1] * dims[2]) throw new Error(`cube: ${values.length} values for a ${dims.join("x")} grid`);
  return { comments: [lines[0], lines[1]], natoms: lines[2], dims, axes, atoms: lines.slice(6, 6 + n), extra, values };
}

const axisLine = (count: number, v: number[]) => `${String(count).padStart(5)}${v.map((x) => x.toFixed(6).padStart(12)).join("")}`;

/** The rows of a grid's values, six per line and a new line for every (x, y) row (Gaussian's). */
function valueLines(values: string[], nz: number): string[] {
  const out: string[] = [];
  for (let r = 0; r < values.length; r += nz) {
    const row = values.slice(r, r + nz);
    for (let k = 0; k < row.length; k += 6) out.push(row.slice(k, k + 6).map((v) => v.padStart(13)).join(""));
  }
  return out;
}

/** The cube with every `n`-th point along each axis (n <= 1: the text unchanged). */
export function strideCube(text: string, n: number): string {
  if (!(n > 1)) return text;
  const c = parseCube(text);
  const [d0, d1, d2] = c.dims;
  const nd = c.dims.map((d) => Math.floor((d - 1) / n) + 1) as [number, number, number];
  const picked: string[] = [];
  for (let i = 0; i < d0; i += n) for (let j = 0; j < d1; j += n) for (let k = 0; k < d2; k += n) picked.push(c.values[(i * d1 + j) * d2 + k]);
  const axes = c.axes.map((a, ax) => axisLine(nd[ax], a.slice(1).map((x) => x * n)));
  const comment = `${c.comments[1]}; grid stride ${n} (scene-export)`;
  return [c.comments[0], comment, c.natoms, ...axes, ...c.atoms, ...c.extra, ...valueLines(picked, nd[2])].join("\n") + "\n";
}
