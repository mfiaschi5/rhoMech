// MolDepiction: one pool or step depiction (RDKit.js) with exact atom positions and, on top,
// the curved-arrow overlay (design §11.4, §11.5).
//
// Props:
//   depiction      DepictionOut | null     from GET /pool/{state}/depiction or a StepViewOut
//                                          (null: a "loading" placeholder of the same size)
//   stateId        string                  written to data-state-id (default: depiction.ref)
//   width, height  number                  pixel size (default 320 x 220)
//   highlightMaps  number[]                atoms to highlight, by map number
//   arrows         TokenArrowLike[]        token arrows to draw 
//   datasetArrows  DatasetArrowOut[]       dataset arrows (map mode)
//   edgeId         string                  data-edge of the arrows
//   onArrowClick   (index) => void         click on a main arrow (opens the physics panel)
//   selectedArrow  number | null           the selected main arrow
//   fadedArrows    boolean                 draw the arrows faded (Correct page: the original edge)
//   fetchServer    () => Promise<DepictionOut>  tier-3 fetcher for refs that are not state ids
//   testId         string                  default "mol-depiction"
//   className      string
//   onRendered     (r: DepictionResult) => void
//   children       ReactNode               extra content inside the container (absolute layers)
//
// DOM: <div data-testid="mol-depiction" data-state-id data-coords-method="metadata|fit|server"
// data-atom-xy='{"<map>": [x, y], ...}' data-render-state="loading|ready|error">: the RDKit SVG,
// then the ArrowOverlay <svg> with the same viewBox placed exactly on top.
import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { DatasetArrowOut, DepictionOut } from "../api/types.gen";
import type { TokenArrowLike } from "../chem/arrowGeom";
import { renderDepiction, type DepictionResult, type Point } from "../chem/depict";
import { ArrowOverlay } from "./ArrowOverlay";

export interface MolDepictionProps {
  depiction: DepictionOut | null;
  stateId?: string;
  width?: number;
  height?: number;
  highlightMaps?: readonly number[];
  arrows?: readonly TokenArrowLike[] | null;
  datasetArrows?: readonly DatasetArrowOut[] | null;
  edgeId?: string;
  onArrowClick?: (index: number) => void;
  selectedArrow?: number | null;
  fadedArrows?: boolean;
  fetchServer?: () => Promise<DepictionOut>;
  testId?: string;
  className?: string;
  onRendered?: (r: DepictionResult) => void;
  children?: ReactNode;
}

function atomXYJson(byMap: ReadonlyMap<number, Point>): string {
  const obj: Record<string, [number, number]> = {};
  for (const [map, [x, y]] of byMap) obj[String(map)] = [Math.round(x * 100) / 100, Math.round(y * 100) / 100];
  return JSON.stringify(obj);
}

export function MolDepiction(props: MolDepictionProps) {
  const { depiction, width = 320, height = 220, highlightMaps, fetchServer, onRendered } = props;
  const [state, setState] = useState<{ result: DepictionResult | null; error: string | null }>({
    result: null,
    error: null,
  });
  const highlight = useMemo(() => {
    if (!depiction || !highlightMaps?.length) return [];
    const want = new Set(highlightMaps);
    return depiction.atoms.filter((a) => want.has(a.map)).map((a) => a.depiction_idx);
  }, [depiction, highlightMaps]);
  const highlightKey = highlight.join(",");

  useEffect(() => {
    let live = true;
    setState({ result: null, error: null });
    if (!depiction) return;
    renderDepiction(depiction, { width, height, highlight, fetchServer }).then(
      (result) => {
        if (!live) return;
        setState({ result, error: null });
        onRendered?.(result);
      },
      (err: unknown) => live && setState({ result: null, error: err instanceof Error ? err.message : String(err) }),
    );
    return () => {
      live = false;
    };
    // highlightKey stands for `highlight`; the callbacks are deliberately not dependencies
  }, [depiction, width, height, highlightKey]);

  const { result, error } = state;
  const renderState = error ? "error" : result ? "ready" : "loading";
  return (
    <div
      data-testid={props.testId ?? "mol-depiction"}
      className={`mol-depiction${props.className ? ` ${props.className}` : ""}`}
      data-state-id={props.stateId ?? depiction?.ref ?? ""}
      data-coords-method={result?.method ?? ""}
      data-atom-xy={result ? atomXYJson(result.byMap) : "{}"}
      data-render-state={renderState}
      style={{ width, height }}
    >
      {result && <div className="mol-svg" dangerouslySetInnerHTML={{ __html: result.svg }} />}
      {result && (
        <ArrowOverlay
          atomXY={result.byMap}
          viewBox={result.viewBox}
          width={width}
          height={height}
          arrows={props.arrows}
          datasetArrows={props.datasetArrows}
          elementOf={result.elements}
          edgeId={props.edgeId}
          onArrowClick={props.onArrowClick}
          selectedIndex={props.selectedArrow}
          faded={props.fadedArrows}
        />
      )}
      {!result && !error && depiction && <div className="mol-placeholder">drawing…</div>}
      {error && (
        <div className="mol-placeholder mol-error" title={error}>
          depiction failed: {error}
        </div>
      )}
      {props.children}
    </div>
  );
}
