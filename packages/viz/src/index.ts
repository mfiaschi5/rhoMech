// @ksmech/viz: the ksmech visualizations as props-only React components (they never fetch, read
// an app store or a router) and the scene format that feeds them from static files. The
// sub-modules (`@ksmech/viz/<dir>/<module>`) are importable too; this is the public surface.
// README.md lists the components, their props and how a static page embeds them.

// ---- scenes: the versioned documents, their validation, loading and view
export { SCENE_KIND, SCENE_TYPES, SCENE_VERSION, STEP_SCENE_VARIANTS } from "./scene/types";
export type { AssetRef, CubeAsset, MoleculeScene, NetworkScene, Scene, SceneDoc, SceneEdge, SceneProvenance, SceneType, StepScene, StepSceneVariant } from "./scene/types";
export { isRelativeAssetPath, validateScene, type ValidationResult } from "./scene/validate";
export { assetLoader, checkAssetBytes, loadScene, NO_ASSETS, SceneError, sha256Hex, type AssetLoader, type LoadedScene, type LoadOptions } from "./scene/load";
export { SceneView, type SceneViewProps } from "./scene/SceneView";
export { firstOrbital, MoleculeSceneView } from "./scene/MoleculeSceneView";
export { cubeLoader, stepVariants, StepSceneView } from "./scene/StepSceneView";
export { NetworkSceneView } from "./scene/NetworkSceneView";

// ---- molecules: 2D (RDKit.js), 3D (3Dmol.js), orbitals
export { MolDepiction, type MolDepictionProps } from "./components/MolDepiction";
export { SmilesDepiction } from "./components/SmilesDepiction";
export { ArrowOverlay, type ArrowOverlayProps } from "./components/ArrowOverlay";
export { renderDepiction, type DepictionResult, type RenderOptions } from "./chem/depict";
export { loadRDKit, smilesToSvg, useRDKit, type RDKitModule } from "./chem/rdkit";
export { probeWebGL, SURFACE_COLORS, threeDmol } from "./chem/threed";
export { Molecule3D, type Molecule3DProps } from "./viewer/Molecule3D";
export { useGLViewer, type ViewerHandle } from "./viewer/useGLViewer";
export { DEFAULT_REP, REPS, type Rep } from "./viewer/rep";
export { LevelDiagram, type LevelDiagramProps } from "./orbitals/LevelDiagram";

// ---- one elementary step: Textbook, Animated, Orbitals
export { StepTextbook, StepCaption, TEXTBOOK_STYLE, type StepVariantProps } from "./step/StepTextbook";
export { StepAnimated } from "./step/StepAnimated";
export { StepOrbital, StepEnergy, ORBITAL_STYLE, type OrbitalData } from "./step/StepOrbital";
export { StepScheme } from "./step/StepScheme";
export { ProductView } from "./step/ProductView";
export { drawnStructure, hoverOf, NO_HOVER, predictedStep, sidesOf, stepSidesOf, useBoxWidth, type HoverSet, type ShownStep, type StepEdge, type StepSides } from "./step/stepSides";
export { stepCaptions, scoreWords, speciesLabel } from "./network/network";

// ---- the mechanism network: Scheme, Pathways
export { SchemeView } from "./network/SchemeView";
export { PathwayExplorer } from "./network/PathwayExplorer";
export { StateCard } from "./network/StateCard";
export { stepInfo, VARIANTS as NETWORK_VARIANTS, type NetViewProps, type Variant as NetworkVariant } from "./network/shared";
export { backEdges, edgeStrength, layeredLayout, pathwayTree, schemeLayout, treeRows } from "./network/netLayout";

// ---- the API types the props use (ksmech-serve's schemas)
export type * as Api from "./apiTypes";
export type { ArrowReadingOut, EdgeOut, MatrixIO, NetworkOut, NodeOut, OrbitalReadingOut, PathwayOut } from "./v9types";
