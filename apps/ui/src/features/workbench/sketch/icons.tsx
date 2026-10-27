// Line icons of the sketcher's action row (18 x 18, stroked in currentColor). Decorative: every
// button that shows one carries its name in title and aria-label.
import type { ReactNode } from "react";

const PATHS = {
  undo: "M6 4 L2.5 7.5 L6 11 M2.5 7.5 H11 A4.5 4.5 0 0 1 11 16.5 H8",
  redo: "M12 4 L15.5 7.5 L12 11 M15.5 7.5 H7 A4.5 4.5 0 0 0 7 16.5 H10",
  "rotate-cw": "M9 15.5 A5.5 5.5 0 1 1 14.5 10 M12.3 8 L14.5 10.4 L16.6 8",
  "rotate-ccw": "M9 15.5 A5.5 5.5 0 1 0 3.5 10 M5.7 8 L3.5 10.4 L1.4 8",
  "flip-h": "M7 5 L2.5 13 H7 Z M11 5 L15.5 13 H11 Z",
  "flip-v": "M5 7 L13 2.5 V7 Z M5 11 L13 15.5 V11 Z",
  fit: "M3 7 V3 H7 M11 3 H15 V7 M15 11 V15 H11 M7 15 H3 V11",
  clear: "M3.5 5 H14.5 M7 5 V3 H11 V5 M5 5 L6 15.5 H12 L13 5",
  import: "M9 2 V11 M5.5 7.5 L9 11 L12.5 7.5 M3 12 V15.5 H15 V12",
  export: "M9 11 V2 M5.5 5.5 L9 2 L12.5 5.5 M3 12 V15.5 H15 V12",
  breaks: "M2 9 H16 M9 2 V6 M9 12 V16",
  tidy: "M1.5 6 H6 V12 H1.5 Z M12 6 H16.5 V12 H12 Z M9 3 V15",
} as const;

export type IconName = keyof typeof PATHS;

/** The mirror axis of the flip icons, dashed. */
const AXIS: Partial<Record<IconName, ReactNode>> = {
  "flip-h": <path d="M9 1.5 V16.5" strokeDasharray="2 1.6" />,
  "flip-v": <path d="M1.5 9 H16.5" strokeDasharray="2 1.6" />,
};

export function Icon({ name }: { name: IconName }) {
  return (
    <svg className="sk-icon" width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      {AXIS[name]}
      <path d={PATHS[name]} />
    </svg>
  );
}
