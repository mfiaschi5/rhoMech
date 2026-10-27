// SceneView: one scene document (scene/types.ts) drawn by the view of its type: a molecule
// (MoleculeSceneView), a step (StepSceneView) or a network (NetworkSceneView), with its title and
// a provenance line. Props only: the document and the loader of its assets (loadScene gives both).
//
// DOM: <div data-testid=<testId, "scene-view"> data-type="molecule|step|network"> with the view
// and <p data-testid="scene-provenance">.
import type { AssetLoader } from "./load";
import { NO_ASSETS } from "./load";
import { MoleculeSceneView } from "./MoleculeSceneView";
import { NetworkSceneView } from "./NetworkSceneView";
import { StepSceneView } from "./StepSceneView";
import type { SceneDoc } from "./types";
import "./scene.css";

export interface SceneViewProps {
  scene: SceneDoc;
  /** The scene's asset files (orbital cubes); loadScene().assets. */
  assets?: AssetLoader;
  testId?: string;
}

function ProvenanceLine({ doc }: { doc: SceneDoc }) {
  const p = doc.provenance;
  const predictor = p.predict?.predictor_version ?? p.predict?.predictor;
  return (
    <p className="muted pv-small scene-provenance" data-testid="scene-provenance">
      {p.server.spec_version} · chemistry {p.server.chemistry} · {p.server.qm_method}
      {predictor ? ` · ${predictor}` : ""} · exported {p.exported_at.slice(0, 10)}
    </p>
  );
}

export function SceneView({ scene, assets = NO_ASSETS, testId = "scene-view" }: SceneViewProps) {
  const s = scene.scene;
  return (
    <div className="scene-view stack" data-testid={testId} data-type={s.type}>
      {s.title && <h3 className="scene-title">{s.title}</h3>}
      {s.type === "molecule" && <MoleculeSceneView scene={s} assets={assets} />}
      {s.type === "step" && <StepSceneView scene={s} assets={assets} />}
      {s.type === "network" && <NetworkSceneView scene={s} />}
      <ProvenanceLine doc={scene} />
    </div>
  );
}
