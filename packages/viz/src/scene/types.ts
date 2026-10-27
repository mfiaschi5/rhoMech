// The scene format, version 1: one JSON document that holds exactly what one view needs, taken
// from the ksmech-serve API responses (the types of apiTypes.ts, narrowed with Pick where a view
// reads less), so a static page (the wiki) draws a molecule, a step or a network without a server.
// Large payloads (orbital cubes) are files next to the scene, referenced by AssetRef: a relative
// path with its byte size and sha256, checked when loaded (load.ts). The JSON Schema of the same
// shape is packages/viz/scene.schema.json; validate.ts checks it by hand.
//
//   { "kind": "ksmech-scene", "version": 1,
//     "scene": { "type": "molecule" | "step" | "network", ... },
//     "provenance": { "exporter", "exported_at", "server", "request", "predict", "notes" } }
import type { DepictionOut, EdgeOut, GeometryOut, MetaOut, MoleculeGeometryOut, MoleculeOut, NetworkOut, OrbitalSetOut, Provenance } from "../apiTypes";

export const SCENE_KIND = "ksmech-scene";
export const SCENE_VERSION = 1;
export const SCENE_TYPES = ["molecule", "step", "network"] as const;
export type SceneType = (typeof SCENE_TYPES)[number];

/** A file next to the scene: `path` is relative to the scene file (no "..", no scheme). */
export interface AssetRef {
  path: string;
  bytes: number;
  /** Lowercase hex sha256 of the file's bytes. */
  sha256: string;
  media_type: string;
}

/** One orbital's Gaussian cube: `index` in the scene's orbital set (molecule) or a token id (step). */
export interface CubeAsset {
  index: number;
  asset: AssetRef;
}

/** A molecule of the workbench (POST /molecule, GET /molecule/{id}/geometry, /orbitals, /cube). */
export interface MoleculeScene {
  type: "molecule";
  title?: string;
  /** The input SMILES (the 2D drawing, RDKit.js). */
  smiles: string;
  molecule: Pick<MoleculeOut, "molecule_id" | "key" | "charge" | "multiplicity" | "chemistry" | "method">;
  geometry: MoleculeGeometryOut;
  /** The canonical orbitals (the level diagram); null without QM orbitals. */
  orbitals: Pick<OrbitalSetOut, "set" | "start" | "source" | "homo_index" | "orbitals"> | null;
  /** Cubes of some of those orbitals (by orbital index). */
  cubes: CubeAsset[];
  /** The orbital shown first (default: the HOMO). */
  selected?: number | null;
}

/** The fields of an edge the step views read. */
export type SceneEdge = Pick<EdgeOut, "id" | "kind" | "source_state_id" | "product_state_id" | "source_mapped_smiles" | "product_mapped_smiles" | "matrix" | "anchor" | "arrows" | "readings" | "score" | "logp" | "predictor" | "rank" | "product_smiles" | "sigma_keys" | "formal_charges">;

export const STEP_SCENE_VARIANTS = ["textbook", "animated", "orbital"] as const;
export type StepSceneVariant = (typeof STEP_SCENE_VARIANTS)[number];

/** One elementary step of a predict job (GET /jobs/{id} edge, GET /pool/{state}/depiction of both
 *  states, /geometry and /orbital/{token} of the source state). */
export interface StepScene {
  type: "step";
  title?: string;
  edge: SceneEdge;
  /** NetworkOut.scores_are_probabilities of the job (log p shown instead of the raw score). */
  probabilities: boolean;
  source: {
    depiction: DepictionOut;
    /** The source pool in 3D (the orbital view); null: no 3D. */
    geometry: GeometryOut | null;
    /** Cubes of the readings' donor and acceptor tokens (CubeAsset.index is the token id). */
    cubes: CubeAsset[];
  };
  /** The product state's depiction; null: the product is drawn from its SMILES. */
  product: { depiction: DepictionOut } | null;
  /** The views offered, the first shown (default: every view the data allows). */
  variants?: StepSceneVariant[];
}

/** The mechanism network of a predict job (GET /jobs/{id} result). */
export interface NetworkScene {
  type: "network";
  title?: string;
  network: NetworkOut;
  /** The view shown first. */
  variant?: "layered" | "explorer";
}

export type Scene = MoleculeScene | StepScene | NetworkScene;

export interface SceneProvenance {
  /** What wrote the file ("ksmech scene-export 1"). */
  exporter: string;
  /** ISO 8601 time of the export. */
  exported_at: string;
  /** The server's versions (GET /api/meta). */
  server: Pick<MetaOut, "api_schema" | "app_version" | "spec_version" | "physics_fingerprint" | "chemistry" | "qm_method">;
  /** What was exported (type, job, edge, smiles, options). */
  request: Record<string, unknown>;
  /** The predict job's provenance (step and network scenes). */
  predict?: Provenance | null;
  notes: string[];
}

export interface SceneDoc<S extends Scene = Scene> {
  kind: typeof SCENE_KIND;
  version: typeof SCENE_VERSION;
  scene: S;
  provenance: SceneProvenance;
}
