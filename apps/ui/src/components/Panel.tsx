// Panel: the Predict and Correct pages' panel frame, styled like the workbench's (a compact
// header with a title, a note and tools that wrap, then the body).
//
// Props: title, note?, tools? (ReactNode, right of the note), testId?, className?, children.
import type { ReactNode } from "react";

export interface PanelProps {
  title: ReactNode;
  note?: ReactNode;
  tools?: ReactNode;
  testId?: string;
  className?: string;
  children?: ReactNode;
}

export function Panel({ title, note, tools, testId, className, children }: PanelProps) {
  return (
    <section className={`panel pv-panel${className ? ` ${className}` : ""}`} data-testid={testId}>
      <header className="pv-head">
        <h3 className="pv-title">{title}</h3>
        {note && <span className="pv-note muted">{note}</span>}
        {tools && <span className="pv-tools">{tools}</span>}
      </header>
      <div className="pv-body">{children}</div>
    </section>
  );
}
