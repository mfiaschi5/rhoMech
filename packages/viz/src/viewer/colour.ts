// Colour scales of the workbench. "rwb" is 3Dmol's Gradient.RWB formula (red at the minimum,
// white in the middle, blue at the maximum, square-root ramps), so the colour bar of the ESP
// matches the surface 3Dmol paints; "seq" is a white-to-blue ramp for unsigned properties.
export type Scale = "rwb" | "seq";

const hex2 = (v: number) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0");
const rgb = (r: number, g: number, b: number) => `#${hex2(r)}${hex2(g)}${hex2(b)}`;

/** The colour of `value` on [lo, hi]; out-of-range values are clamped. */
export function colourAt(scale: Scale, value: number, lo: number, hi: number): string {
  if (!Number.isFinite(value) || hi <= lo) return "#bbbbbb";
  const t = Math.max(0, Math.min(1, (value - lo) / (hi - lo)));
  if (scale === "seq") {
    // #f4f8fc -> #6baed6 -> #08306b
    if (t < 0.5) {
      const u = t / 0.5;
      return rgb(244 + (107 - 244) * u, 248 + (174 - 248) * u, 252 + (214 - 252) * u);
    }
    const u = (t - 0.5) / 0.5;
    return rgb(107 + (8 - 107) * u, 174 + (48 - 174) * u, 214 + (107 - 214) * u);
  }
  if (t < 0.5) {
    const s = 255 * Math.sqrt(t / 0.5);
    return rgb(255, s, s);
  }
  if (t > 0.5) {
    const s = 255 * Math.sqrt(1 - (t - 0.5) / 0.5);
    return rgb(s, s, 255);
  }
  return "#ffffff";
}

/** A CSS linear-gradient of the scale, low on the left. */
export function cssGradient(scale: Scale, steps = 12): string {
  const stops = Array.from({ length: steps + 1 }, (_, i) => {
    const t = i / steps;
    return `${colourAt(scale, t, 0, 1)} ${(t * 100).toFixed(1)}%`;
  });
  return `linear-gradient(to right, ${stops.join(", ")})`;
}
