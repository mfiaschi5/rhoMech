// Feature seam (design §11.1): registers the "wiki" page (the Wiki section of the nav). Imports
// no other feature; the scene figures use the shared components/SceneEmbed.tsx.
import { registerPage } from "../../registry";
import { WikiPage } from "./WikiPage";
import "./wiki.css";

registerPage("wiki", WikiPage);
