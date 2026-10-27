// Window global defined by the classic script /vendor/RDKit_minimal.js (index.html).
// The types come from the package's own declarations; nothing from @rdkit/rdkit is bundled.
import type { MainModule } from "@rdkit/rdkit";

export interface RDKitInitOptions {
  /** Maps a runtime file name (RDKit_minimal.wasm) to its URL. */
  locateFile?: (file: string, prefix?: string) => string;
}

declare global {
  interface Window {
    initRDKitModule?: (options?: RDKitInitOptions) => Promise<MainModule>;
  }
}
