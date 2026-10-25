// The contact placement scene of the predict 3D panel (design §8.3, §11.7), drawn with 3Dmol:
// the donor fragment, the placed acceptor cap (alternate colours, opacity 0.8), the faded
// acceptor context, the two lobe arrows (u at the donor site a, -u at the placed acceptor site b,
// 1.2 Angstrom long) and the dashed a-b cylinder labelled with the contact distance. The
// generic helpers (WebGL probe, cube parsing, isosurfaces) are in chem/threed.ts.
import type { GLViewer } from "3dmol";
import type { ContactOut, LobeOut } from "../../api/types.gen";
import { SURFACE_COLORS } from "../../chem/threed";

export const LOBE_LEN_ANG = 1.2;

type Vec = [number, number, number];
const xyz = (v: readonly number[]) => ({ x: v[0], y: v[1], z: v[2] });
const along = (o: readonly number[], d: readonly number[], len: number): Vec => [o[0] + len * d[0], o[1] + len * d[1], o[2] + len * d[2]];

function addLobe(viewer: GLViewer, lobe: LobeOut, color: string, radius: number): void {
  viewer.addArrow({ start: xyz(lobe.origin), end: xyz(along(lobe.origin, lobe.direction, LOBE_LEN_ANG)), radius, radiusRatio: 2.2, mid: 0.72, color });
}

/** Models, lobes and the a-b distance of one contact placement. */
export function drawContact(viewer: GLViewer, c: ContactOut): void {
  const donor = viewer.addModel(c.donor_fragment.molblock, "sdf");
  donor.setStyle({}, { stick: { radius: 0.14 }, sphere: { scale: 0.22 } });
  const cap = viewer.addModel(c.cap.molblock, "sdf");
  cap.setStyle({}, { stick: { radius: 0.14, colorscheme: "greenCarbon", opacity: 0.8 }, sphere: { scale: 0.22, colorscheme: "greenCarbon", opacity: 0.8 } });
  if (c.acceptor_context) {
    const ctx = viewer.addModel(c.acceptor_context.molblock, "sdf");
    ctx.setStyle({}, { stick: { radius: 0.08, colorscheme: "greenCarbon", opacity: 0.25 } });
  }
  addLobe(viewer, c.donor_lobe, SURFACE_COLORS.donor.pos, 0.07);
  for (const t of c.donor_lobe.tied) addLobe(viewer, { ...c.donor_lobe, direction: t }, "#8aa6d8", 0.035);
  addLobe(viewer, c.acceptor_lobe_placed, SURFACE_COLORS.acceptor.pos, 0.07);
  const a = c.donor_lobe.origin;
  const b = c.acceptor_lobe_placed.origin;
  viewer.addCylinder({ start: xyz(a), end: xyz(b), radius: 0.03, dashed: true, dashLength: 0.12, gapLength: 0.1, color: "#555555", fromCap: "none", toCap: "none" });
  const mid: Vec = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
  viewer.addLabel(`${c.distance_ang.toFixed(2)} Å`, { position: xyz(mid), fontSize: 12, fontColor: "black", backgroundColor: "white", backgroundOpacity: 0.75, inFront: true });
}
