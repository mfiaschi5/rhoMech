// StateDepiction: MolDepiction of one saved pool state (GET /pool/{state}/depiction), keeping
// the explicit H atoms the arrows need, with the arrows drawn on top.
//
// Props:
//   stateId        string | null
//   arrows         TokenArrowLike[]        drawn with ArrowOverlay (main + implied)
//   edgeId         string                  data-edge of the arrows
//   onArrowClick   (index) => void         opens the physics drawer
//   selectedArrow  number | null
//   fadedArrows    boolean
//   extraMaps      number[]                more map numbers to keep (H atoms)
//   width, height  number                  default 300 x 200
//   caption        ReactNode               shown under the drawing
//   testId         string                  data-testid of the wrapper (optional)
import type { ReactNode } from "react";
import type { TokenArrowLike } from "../../chem/arrowGeom";
import { MolDepiction } from "../../components/MolDepiction";
import { usePoolDepiction } from "./hooks";
import { arrowMaps } from "./network";

export interface StateDepictionProps {
  stateId: string | null;
  arrows?: readonly TokenArrowLike[];
  edgeId?: string;
  onArrowClick?: (index: number) => void;
  selectedArrow?: number | null;
  fadedArrows?: boolean;
  extraMaps?: readonly number[];
  width?: number;
  height?: number;
  caption?: ReactNode;
  testId?: string;
  className?: string;
}

export function StateDepiction(props: StateDepictionProps) {
  const { stateId, arrows = [], width = 300, height = 200 } = props;
  const maps = [...arrowMaps(arrows), ...(props.extraMaps ?? [])];
  const dep = usePoolDepiction(stateId, maps);
  return (
    <figure className={`state-depiction${props.className ? ` ${props.className}` : ""}`} data-testid={props.testId}>
      {dep.error ? (
        <div className="mol-depiction mol-failed" style={{ width, height }} data-testid="mol-depiction" data-state-id={stateId ?? ""} data-render-state="error">
          <div className="mol-placeholder mol-error" title={dep.error.message}>
            depiction failed: {dep.error.code}
          </div>
        </div>
      ) : (
        <MolDepiction
          depiction={dep.data ?? null}
          stateId={stateId ?? undefined}
          width={width}
          height={height}
          arrows={arrows}
          edgeId={props.edgeId}
          onArrowClick={props.onArrowClick}
          selectedArrow={props.selectedArrow}
          fadedArrows={props.fadedArrows}
        />
      )}
      {props.caption && <figcaption className="muted">{props.caption}</figcaption>}
    </figure>
  );
}
