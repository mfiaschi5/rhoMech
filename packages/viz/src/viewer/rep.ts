// The 3D representations of a molecule (the 3Dmol style of viewer/scene.ts's repStyle).
export const REPS = ["ball-and-stick", "sticks", "spacefill", "wireframe", "lines"] as const;
export type Rep = (typeof REPS)[number];
export const DEFAULT_REP: Rep = "ball-and-stick";
