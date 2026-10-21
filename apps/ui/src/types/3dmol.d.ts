// Window global defined by the classic script /vendor/3Dmol-min.js (index.html), which sets
// window.$3Dmol. Use it as `window.$3Dmol?.createViewer(element, {...})`; nothing is bundled.
type ThreeDmolNamespace = typeof import("3dmol");

declare global {
  interface Window {
    $3Dmol?: ThreeDmolNamespace;
  }
}

export {};
