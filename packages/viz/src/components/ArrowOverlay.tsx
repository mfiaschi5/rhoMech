// ArrowOverlay: curved electron-pushing arrows drawn as an SVG layer over a depiction
// (design §11.5). MolDepiction renders it for you; use it directly only over a custom drawing.
//
// Props:
//   atomXY        ReadonlyMap<map number, [x, y]>  atom positions in viewBox units (required)
//   viewBox       [x, y, w, h]                     the depiction's viewBox (required)
//   width, height number                           pixel size, equal to the depiction's (required)
//   arrows        TokenArrowLike[]                 token arrows:
//                                                  each gives a "main" arrow, plus an "implied" sink
//                                                  arrow when sink_partner_map is set
//   datasetArrows DatasetArrowOut[]                dataset arrows in map mode ("dataset")
//   elementOf     ReadonlyMap<map, element>        element symbols for the dataset arrows' data attrs
//   edgeId        string                           written to data-edge on every arrow
//   onArrowClick  (index) => void                  click / Enter on a main arrow (hit path or handle)
//   selectedIndex number | null                    the main arrow drawn as selected
//   faded         boolean                          draw every arrow faded (e.g. the original edge)
//   showImplied   boolean (default true)
//
// DOM, per arrow: <g data-testid="arrow-overlay" data-role data-edge data-index data-donor-site
// data-acceptor-site data-donor-site-map data-acceptor-site-map data-donor-element
// data-acceptor-element data-x1 data-y1 data-x2 data-y2 data-mx data-my>, holding (main only) a
// transparent 14 px path.arrow-hit with pointer-events: stroke, the visible path.arrow-line with
// its head-full / head-half marker, and (main only) circle[data-testid=arrow-handle] r=7 at t=0.5.
// An arrow whose atoms have no position is not drawn: the layer <svg> gets data-overlay-error.
import { useEffect, useId, useMemo, type KeyboardEvent } from "react";
import type { DatasetArrowOut } from "../apiTypes";
import {
  centroidOf,
  layoutDatasetArrows,
  layoutTokenArrows,
  type ArrowRole,
  type OverlayItem,
  type TokenArrowLike,
} from "../chem/arrowGeom";
import type { Point, ViewBox } from "../chem/depict";

export interface ArrowOverlayProps {
  atomXY: ReadonlyMap<number, Point>;
  viewBox: ViewBox;
  width: number;
  height: number;
  arrows?: readonly TokenArrowLike[] | null;
  datasetArrows?: readonly DatasetArrowOut[] | null;
  elementOf?: ReadonlyMap<number, string>;
  edgeId?: string;
  onArrowClick?: (index: number) => void;
  selectedIndex?: number | null;
  faded?: boolean;
  showImplied?: boolean;
}

const ROLES: ArrowRole[] = ["main", "implied", "dataset"];

function Markers({ uid }: { uid: string }) {
  return (
    <defs>
      {ROLES.map((role) => (
        <g key={role}>
          <marker
            id={`${uid}-head-full-${role}`}
            className={`head-full marker-${role}`}
            data-marker="head-full"
            viewBox="0 0 9 8"
            markerWidth={9}
            markerHeight={8}
            refX={9}
            refY={4}
            orient="auto"
            markerUnits="userSpaceOnUse"
          >
            <path d="M 0 0 L 9 4 L 0 8 z" />
          </marker>
          <marker
            id={`${uid}-head-half-${role}`}
            className={`head-half marker-${role}`}
            data-marker="head-half"
            viewBox="0 0 9 8"
            markerWidth={9}
            markerHeight={8}
            refX={9}
            refY={4}
            orient="auto"
            markerUnits="userSpaceOnUse"
          >
            <path d="M 0 0 L 9 4 L 1.5 4 z" />
          </marker>
        </g>
      ))}
    </defs>
  );
}

function dataAttrs(item: OverlayItem, edgeId: string | undefined): Record<string, string> {
  const out: Record<string, string> = {
    "data-role": item.role,
    "data-index": String(item.index),
    "data-edge": edgeId ?? "",
  };
  for (const [k, v] of Object.entries(item.data)) out[`data-${k}`] = v;
  return out;
}

export function ArrowOverlay(props: ArrowOverlayProps) {
  const uid = useId().replace(/[^A-Za-z0-9_-]/g, "");
  const { atomXY, arrows, datasetArrows, elementOf, showImplied = true } = props;
  const layout = useMemo(() => {
    const pos = (m: number) => atomXY.get(m);
    const centroid = centroidOf(atomXY.values());
    const token = layoutTokenArrows(arrows ?? [], pos, centroid);
    const dataset = layoutDatasetArrows(datasetArrows ?? [], pos, centroid, (m) => elementOf?.get(m));
    const items = [...token.items, ...dataset.items].filter((i) => showImplied || i.role !== "implied");
    return { items, missing: [...token.missing, ...dataset.missing] };
  }, [atomXY, arrows, datasetArrows, elementOf, showImplied]);

  const error = layout.missing.length
    ? layout.missing.map((m) => `${m.role} arrow ${m.index}: no position for map ${m.maps.join(",") || "?"}`).join("; ")
    : undefined;
  useEffect(() => {
    if (error) console.warn(`ArrowOverlay: ${error}`);
  }, [error]);

  const click = (index: number) => props.onArrowClick?.(index);
  const onKey = (e: KeyboardEvent, index: number) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      click(index);
    }
  };
  const [vx, vy, vw, vh] = props.viewBox;
  return (
    <svg
      className={`arrow-overlay-layer${props.faded ? " faded" : ""}`}
      data-testid="arrow-overlay-layer"
      data-overlay-error={error}
      viewBox={`${vx} ${vy} ${vw} ${vh}`}
      width={props.width}
      height={props.height}
      preserveAspectRatio="xMidYMid meet"
    >
      <Markers uid={uid} />
      {layout.items.map((item) => {
        const main = item.role === "main";
        const selected = main && props.selectedIndex === item.index;
        const head = item.electrons === 1 ? "head-half" : "head-full";
        return (
          <g
            key={item.key}
            data-testid="arrow-overlay"
            className={`arrow arrow-${item.role}${selected ? " selected" : ""}`}
            {...dataAttrs(item, props.edgeId)}
          >
            {main && <path className="arrow-hit" d={item.curve.d} onClick={() => click(item.index)} />}
            <path className="arrow-line" d={item.curve.d} markerEnd={`url(#${uid}-${head}-${item.role})`} />
            {main && (
              <circle
                data-testid="arrow-handle"
                className="arrow-handle"
                cx={item.curve.mid[0]}
                cy={item.curve.mid[1]}
                r={7}
                role="button"
                tabIndex={0}
                aria-label={`arrow ${item.index + 1}: ${item.data["donor-element"]} to ${item.data["acceptor-element"]}`}
                onClick={() => click(item.index)}
                onKeyDown={(e) => onKey(e, item.index)}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
