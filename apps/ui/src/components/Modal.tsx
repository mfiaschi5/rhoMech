// Modal: a centred dialog over a dimmed page; Escape and the backdrop close it.
//
// Props: title, onClose, testId, wide? (up to 1100 px instead of 760), children, footer?.
// DOM: <div class="pv-modal-backdrop"><div role="dialog" aria-modal data-testid={testId}> with a
// header (the title, a close button <testId>-close), the body and the footer.
import { useEffect, type ReactNode } from "react";

export interface ModalProps {
  title: ReactNode;
  onClose: () => void;
  testId: string;
  wide?: boolean;
  children?: ReactNode;
  footer?: ReactNode;
}

export function Modal({ title, onClose, testId, wide, children, footer }: ModalProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // the sketcher's own Escape (cancel a gesture) stays inside it
      if (e.key === "Escape" && !(e.target as Element | null)?.closest?.("[data-testid=sketch]")) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="pv-modal-backdrop" onPointerDown={(e) => e.target === e.currentTarget && onClose()}>
      <div role="dialog" aria-modal="true" className={`pv-modal panel${wide ? " pv-modal-wide" : ""}`} data-testid={testId}>
        <header className="pv-head">
          <h3 className="pv-title">{title}</h3>
          <span className="pv-tools">
            <button type="button" className="btn btn-small" data-testid={`${testId}-close`} onClick={onClose} aria-label="close">
              ✕
            </button>
          </span>
        </header>
        <div className="pv-modal-body">{children}</div>
        {footer && <footer className="row pv-modal-footer">{footer}</footer>}
      </div>
    </div>
  );
}
