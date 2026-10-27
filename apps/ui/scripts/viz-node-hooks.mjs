// `node --import ./scripts/viz-node-hooks.mjs ...` (npm test, npm run scene:export): resolve @ksmech/viz
// under node the way Vite does (scripts/viz-alias.mjs).
import { register } from "node:module";

register("./viz-alias.mjs", import.meta.url);
