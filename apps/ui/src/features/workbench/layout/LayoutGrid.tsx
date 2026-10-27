// LayoutGrid: the workbench panels placed by the layout state (layoutState.ts): the columns side
// by side with a drag splitter between each pair, then the full-width row; every panel in a
// PanelFrame whose header collapses, hides and moves it, with a row splitter under the panels
// that have a height (sketch canvas, 3D view, orbital diagram, property tables). Hidden and
// collapsed panels stay mounted (display: none), so the 3D view and the tables keep their
// state. Below 1000 px the columns stack and the column splitters go (workbench.css).
//
// DOM: <div data-testid="wb-grid" data-columns> with <div data-testid="wb-col-<k>"> and
// wb-splitter-col-<k> (between visible columns k-1 and k), <div data-testid="wb-wide">; each
// panel <section class="panel" data-testid="wb-panel-<id>" data-collapsed data-hidden
// data-area="col-<k>|wide"> with header wb-panel-head-<id>: panel-collapse-<id>
// [aria-expanded], panel-move-<id>-<left|right|up|down> (disabled where it cannot move),
// panel-hide-<id>; and wb-splitter-row-<id> under a panel with a height.
import { useRef, type CSSProperties, type Dispatch, type ReactNode } from "react";
import {
  HEIGHTS,
  PANEL_TITLES,
  canMove,
  gridTemplate,
  isHeightPanel,
  visibleColumns,
  type Direction,
  type LayoutAction,
  type LayoutState,
  type PanelId,
} from "./layoutState";
import { Splitter } from "./Splitter";

export interface PanelContent {
  body: ReactNode;
  /** Short status next to the title (e.g. "not computed"). */
  note?: ReactNode;
}

export interface LayoutGridProps {
  layout: LayoutState;
  dispatch: Dispatch<LayoutAction>;
  panels: Record<PanelId, PanelContent>;
}

const MOVES: { dir: Direction; glyph: string; label: string }[] = [
  { dir: "left", glyph: "←", label: "Move left" },
  { dir: "right", glyph: "→", label: "Move right" },
  { dir: "up", glyph: "↑", label: "Move up" },
  { dir: "down", glyph: "↓", label: "Move down" },
];

/** The CSS variable of a panel's height (read by the panel's own styles). */
const HEIGHT_VAR: Record<string, string> = { sketch: "--wb-sketch-h", viewer: "--wb-viewer-h", orbitals: "--wb-orbitals-h", properties: "--wb-table-h" };

function PanelHead({ id, layout, dispatch, note }: { id: PanelId; layout: LayoutState; dispatch: Dispatch<LayoutAction>; note?: ReactNode }) {
  const collapsed = layout.collapsed.includes(id);
  const title = PANEL_TITLES[id];
  return (
    <header className="wb-panel-head" data-testid={`wb-panel-head-${id}`}>
      <button
        type="button"
        className="wb-panel-title"
        data-testid={`panel-collapse-${id}`}
        aria-expanded={!collapsed}
        title={collapsed ? `Expand ${title}` : `Collapse ${title}`}
        onClick={() => dispatch({ type: "collapse", panel: id })}
      >
        <span className="wb-caret" aria-hidden="true">{collapsed ? "▸" : "▾"}</span>
        {title}
      </button>
      {note && <span className="wb-panel-note muted">{note}</span>}
      <span className="wb-spacer" />
      <span className="wb-panel-tools" role="group" aria-label={`${title} panel`}>
        {MOVES.map((m) => (
          <button key={m.dir} type="button" className="wb-panel-tool" data-testid={`panel-move-${id}-${m.dir}`} disabled={!canMove(layout, id, m.dir)} title={`${m.label}: ${title}`} aria-label={`${m.label}: ${title}`} onClick={() => dispatch({ type: "move", panel: id, dir: m.dir })}>
            {m.glyph}
          </button>
        ))}
        <button type="button" className="wb-panel-tool" data-testid={`panel-hide-${id}`} title={`Hide ${title} (the Layout menu shows it again)`} aria-label={`Hide ${title}`} onClick={() => dispatch({ type: "hide", panel: id })}>
          ×
        </button>
      </span>
    </header>
  );
}

function PanelFrame({ id, area, layout, dispatch, content }: { id: PanelId; area: string; layout: LayoutState; dispatch: Dispatch<LayoutAction>; content: PanelContent }) {
  const collapsed = layout.collapsed.includes(id);
  const hidden = layout.hidden.includes(id);
  const h = isHeightPanel(id) ? layout.heights[id] : null;
  const style = h !== null ? ({ [HEIGHT_VAR[id]]: `${h}px` } as CSSProperties) : undefined;
  const from = useRef(h ?? 0);
  return (
    <section className={`panel wb-panel wb-panel-${id}`} data-testid={`wb-panel-${id}`} data-panel={id} data-area={area} data-collapsed={collapsed ? "true" : "false"} data-hidden={hidden ? "true" : "false"} hidden={hidden} style={style}>
      <PanelHead id={id} layout={layout} dispatch={dispatch} note={content.note} />
      <div className="wb-panel-body" hidden={collapsed}>
        {content.body}
      </div>
      {h !== null && isHeightPanel(id) && !collapsed && (
        <Splitter
          testId={`wb-splitter-row-${id}`}
          orientation="horizontal"
          label={`Resize ${PANEL_TITLES[id]} (${HEIGHTS[id].min}-${HEIGHTS[id].max} px)`}
          onStart={() => (from.current = layout.heights[id])}
          onDrag={(d) => dispatch({ type: "setHeight", panel: id, px: from.current + d })}
        />
      )}
    </section>
  );
}

/** The splitter between visible columns `left` and `right` (indices into layout.columns). */
function ColumnSplitter({ k, left, right, layout, dispatch, gridRef }: { k: number; left: number; right: number; layout: LayoutState; dispatch: Dispatch<LayoutAction>; gridRef: React.RefObject<HTMLDivElement | null> }) {
  const from = useRef<{ pair: [number, number]; perPx: number }>({ pair: [0, 0], perPx: 0 });
  const start = () => {
    const shown = visibleColumns(layout).reduce((s, c) => s + layout.widths[c.index], 0);
    const width = gridRef.current?.getBoundingClientRect().width ?? 1000;
    from.current = { pair: [layout.widths[left], layout.widths[right]], perPx: shown / Math.max(1, width) };
  };
  return (
    <Splitter
      testId={`wb-splitter-col-${k}`}
      orientation="vertical"
      label="Resize the columns"
      onStart={start}
      onDrag={(d) => dispatch({ type: "resizeColumns", left, right, from: from.current.pair, delta: d * from.current.perPx })}
    />
  );
}

export function LayoutGrid({ layout, dispatch, panels }: LayoutGridProps) {
  const gridRef = useRef<HTMLDivElement>(null);
  const shown = visibleColumns(layout);
  const shownIdx = new Set(shown.map((c) => c.index));
  const frame = (id: PanelId, area: string) => <PanelFrame key={id} id={id} area={area} layout={layout} dispatch={dispatch} content={panels[id]} />;
  let k = -1;
  return (
    <>
      <div className="wb-grid" ref={gridRef} data-testid="wb-grid" data-columns={shown.length} style={{ "--wb-cols": gridTemplate(shown) } as CSSProperties}>
        {layout.columns.map((col, i) => {
          const visible = shownIdx.has(i);
          if (visible) k += 1;
          const prev = visible && k > 0 ? shown[k - 1].index : null;
          return [
            prev !== null && <ColumnSplitter key={`s${i}`} k={k} left={prev} right={i} layout={layout} dispatch={dispatch} gridRef={gridRef} />,
            <div key={`c${i}`} className="wb-col" data-testid={visible ? `wb-col-${k}` : undefined} hidden={!visible}>
              {col.map((id) => frame(id, `col-${i}`))}
            </div>,
          ];
        })}
      </div>
      <div className="wb-wide" data-testid="wb-wide">
        {layout.wide.map((id) => frame(id, "wide"))}
      </div>
    </>
  );
}
