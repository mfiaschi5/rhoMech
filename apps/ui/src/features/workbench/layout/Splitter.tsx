// Splitter: a drag handle between two columns ("vertical", drags sideways) or under a panel
// ("horizontal", drags up and down). It reports the pointer's offset in px from where the drag
// began (onDrag) and when it ends; the arrow keys step by 16 px (Shift: 64) as one drag each.
//
// DOM: <div role="separator" data-testid=<testId> aria-orientation data-dragging> (focusable).
import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

export interface SplitterProps {
  testId: string;
  orientation: "vertical" | "horizontal";
  label: string;
  /** A drag starts (before the first onDrag). */
  onStart(): void;
  /** Offset (px) of the pointer since the start, along the drag axis. */
  onDrag(delta: number): void;
}

const KEY_STEP = 16;

export function Splitter({ testId, orientation, label, onStart, onDrag }: SplitterProps) {
  const start = useRef<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const vertical = orientation === "vertical";
  const pos = (e: PointerEvent) => (vertical ? e.clientX : e.clientY);
  const down = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    start.current = pos(e);
    setDragging(true);
    onStart();
  };
  const move = (e: PointerEvent<HTMLDivElement>) => {
    if (start.current !== null) onDrag(pos(e) - start.current);
  };
  const up = () => {
    start.current = null;
    setDragging(false);
  };
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    const back = vertical ? "ArrowLeft" : "ArrowUp";
    const fwd = vertical ? "ArrowRight" : "ArrowDown";
    if (e.key !== back && e.key !== fwd) return;
    e.preventDefault();
    onStart();
    onDrag((e.key === fwd ? 1 : -1) * KEY_STEP * (e.shiftKey ? 4 : 1));
  };
  return (
    <div
      role="separator"
      tabIndex={0}
      aria-orientation={orientation}
      aria-label={label}
      title={`${label} (drag, or arrow keys)`}
      data-testid={testId}
      data-dragging={dragging ? "true" : "false"}
      className={`wb-splitter wb-splitter-${vertical ? "col" : "row"}`}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
      onKeyDown={key}
    />
  );
}
