// Entry point. Every feature (src/features/<name>/index.ts[x]) is imported eagerly for its
// registerPage() side effects, so this file never names a feature and builds without any.
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { loadRDKit } from "./chem/rdkit";
import "./styles.css";

const features = import.meta.glob("./features/*/index.{ts,tsx}", { eager: true });
document.body.dataset.features = Object.keys(features)
  .map((path) => path.split("/")[2])
  .sort()
  .join(",");

// Start the wasm download at once; body[data-rdkit] becomes "ready" (or "error").
document.body.dataset.rdkit = "loading";
loadRDKit().catch(() => undefined);

const root = document.getElementById("root");
if (!root) throw new Error("index.html has no #root element");
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
