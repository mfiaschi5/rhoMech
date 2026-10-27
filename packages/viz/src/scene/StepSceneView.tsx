// StepSceneView: a step scene (scene/types.ts StepScene) in the step views of the Predict page,
// Textbook, Animated and Orbitals (StepTextbook, StepAnimated, StepOrbital), switched by a
// segmented control; the orbital view reads its cubes through the scene's asset loader.
//
// DOM: <section data-testid="scene-step" data-variant data-edge> with
// button[data-testid=scene-step-variant-<v>][aria-pressed] and the variant's view inside
// <div data-testid="scene-step-view" data-variant> (its ids prefixed "scene-step").
import { useMemo, useState } from "react";
import { stepCaptions } from "../network/network";
import { StepAnimated } from "../step/StepAnimated";
import { StepOrbital, type OrbitalData } from "../step/StepOrbital";
import { NO_HOVER, predictedStep, stepSidesOf, type HoverSet } from "../step/stepSides";
import { StepTextbook, type StepVariantProps } from "../step/StepTextbook";
import type { AssetLoader } from "./load";
import { STEP_SCENE_VARIANTS, type CubeAsset, type StepScene, type StepSceneVariant } from "./types";

const TITLES: Record<StepSceneVariant, string> = { textbook: "Textbook", animated: "Animated", orbital: "Orbitals" };

/** The views a scene can show: its own list, without the orbital view when it has no 3D. */
export function stepVariants(scene: StepScene): StepSceneVariant[] {
  const can = (v: StepSceneVariant) => v !== "orbital" || scene.source.geometry !== null;
  const list = (scene.variants?.length ? scene.variants : [...STEP_SCENE_VARIANTS]).filter(can);
  return list.length ? list : ["textbook"];
}

/** A token's cube from the scene's assets (a token without one fails with the reason). */
export function cubeLoader(cubes: readonly CubeAsset[], assets: AssetLoader): (token: number) => Promise<string> {
  return (token) => {
    const cube = cubes.find((c) => c.index === token);
    return cube ? assets(cube.asset) : Promise.reject(new Error(`the scene has no cube of token ${token}`));
  };
}

function VariantTabs({ list, value, onChange }: { list: StepSceneVariant[]; value: StepSceneVariant; onChange: (v: StepSceneVariant) => void }) {
  if (list.length < 2) return null;
  return (
    <span className="ss-seg" role="group" aria-label="step view">
      {list.map((v) => (
        <button key={v} type="button" className={`ss-seg-btn${v === value ? " ss-seg-on" : ""}`} aria-pressed={v === value} data-testid={`scene-step-variant-${v}`} onClick={() => onChange(v)}>
          {TITLES[v]}
        </button>
      ))}
    </span>
  );
}

export function StepSceneView({ scene, assets }: { scene: StepScene; assets: AssetLoader }) {
  const list = stepVariants(scene);
  const [variant, setVariant] = useState<StepSceneVariant>(list[0]);
  const [hover, setHover] = useState<HoverSet>(NO_HOVER);
  const step = useMemo(() => predictedStep(scene.edge), [scene.edge]);
  const sides = useMemo(() => stepSidesOf(scene.source.depiction, scene.product?.depiction ?? null, scene.edge, step), [scene, step]);
  const orbital = useMemo((): OrbitalData => ({ key: scene.edge.source_state_id, geometry: scene.source.geometry ?? undefined, depiction: scene.source.depiction, loadCube: cubeLoader(scene.source.cubes, assets) }), [scene, assets]);
  const vp: StepVariantProps = {
    sides,
    step,
    prefix: "scene-step",
    productSmiles: scene.edge.product_smiles,
    sigmaKeys: scene.edge.sigma_keys,
    label: (i) => sides.structure?.label(i) ?? String(i),
    ...stepCaptions(scene.edge, step, scene.probabilities),
    hover,
    onHover: (h) => setHover(h ?? NO_HOVER),
  };
  return (
    <section className="scene-step" data-testid="scene-step" data-variant={variant} data-edge={scene.edge.id}>
      <VariantTabs list={list} value={variant} onChange={setVariant} />
      <div className="ss-view" data-testid="scene-step-view" data-variant={variant}>
        {variant === "textbook" && <StepTextbook {...vp} />}
        {variant === "animated" && <StepAnimated {...vp} />}
        {variant === "orbital" && <StepOrbital {...vp} orbital={orbital} />}
      </div>
    </section>
  );
}
