// MoleculeSceneView: a molecule scene (scene/types.ts MoleculeScene): the 2D drawing of its SMILES
// (RDKit.js), the molecule in 3D with the selected orbital's surfaces (when the scene carries its
// cube) and the canonical level diagram (LevelDiagram of the workbench) to pick an orbital.
//
// DOM: <section data-testid="scene-molecule" data-selected> with scene-molecule-2d
// (SmilesDepiction), scene-molecule-3d (Molecule3D, data-surface), scene-molecule-orbital
// [data-cube="none|loading|ready|error"], scene-molecule-iso and scene-molecule-levels.
import { useState } from "react";
import { SmilesDepiction } from "../components/SmilesDepiction";
import { fmt } from "../format";
import { LevelDiagram } from "../orbitals/LevelDiagram";
import type { Window } from "../orbitals/levels";
import { ISO_MAX, ISO_MIN, ISO_STEP } from "../chem/threed";
import { useLoaded } from "../useLoaded";
import { Molecule3D } from "../viewer/Molecule3D";
import type { AssetLoader } from "./load";
import type { MoleculeScene } from "./types";

/** The orbital shown first: the scene's choice, else the HOMO, else the first with a cube. */
export function firstOrbital(scene: MoleculeScene): number | null {
  return scene.selected ?? scene.orbitals?.homo_index ?? scene.cubes[0]?.index ?? null;
}

function cubeState(has: boolean, cube: { data: string | undefined; error: string | null }): string {
  return !has ? "none" : cube.error ? "error" : cube.data ? "ready" : "loading";
}

function OrbitalLine({ scene, index, state }: { scene: MoleculeScene; index: number | null; state: string }) {
  const row = scene.orbitals?.orbitals.find((r) => r.index === index);
  return (
    <div className="muted pv-small" data-testid="scene-molecule-orbital" data-index={index ?? ""} data-cube={state}>
      {row ? `${row.label ?? `MO ${row.index}`} · ${fmt(row.energy_ev, 2)} eV · occupation ${fmt(row.occupation, 1)}` : "no orbital selected"}
      {state === "none" && row && " · no cube in this scene"}
      {state === "loading" && " · loading the cube…"}
    </div>
  );
}

export function MoleculeSceneView({ scene, assets }: { scene: MoleculeScene; assets: AssetLoader }) {
  const [selected, setSelected] = useState<number | null>(() => firstOrbital(scene));
  const [iso, setIso] = useState(0.05);
  const [win, setWin] = useState<Window>("frontier");
  const cubeRef = scene.cubes.find((c) => c.index === selected);
  const cube = useLoaded(cubeRef ? `${scene.molecule.molecule_id}/${cubeRef.index}` : null, () => assets(cubeRef!.asset));
  const row = scene.orbitals?.orbitals.find((r) => r.index === selected);
  return (
    <section className="scene-molecule" data-testid="scene-molecule" data-selected={selected ?? ""}>
      <div className="scene-molecule-grid">
        <SmilesDepiction smiles={scene.smiles} width={240} height={200} testId="scene-molecule-2d" />
        <div className="stack">
          <Molecule3D geometry={scene.geometry} cube={cube.data ?? null} occupied={(row?.occupation ?? 0) > 1} iso={iso} testId="scene-molecule-3d" />
          <OrbitalLine scene={scene} index={selected} state={cubeState(!!cubeRef, cube)} />
          <label className="scene-iso pv-small">
            iso <input type="range" data-testid="scene-molecule-iso" min={ISO_MIN} max={ISO_MAX} step={ISO_STEP} value={iso} onChange={(e) => setIso(Number(e.target.value))} /> <code className="code">{iso.toFixed(3)}</code>
          </label>
        </div>
        {scene.orbitals && (
          <div data-testid="scene-molecule-levels">
            <LevelDiagram canonical={scene.orbitals.orbitals} lmos={null} selected={selected} window={win} onWindow={setWin} onPickCanonical={setSelected} onPickLmo={setSelected} height={360} />
          </div>
        )}
      </div>
    </section>
  );
}
