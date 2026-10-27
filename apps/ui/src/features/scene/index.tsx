// The scene viewer: #/scene?src=... (an embed page: no nav, no /api/meta).
import { registerPage } from "../../registry";
import { ScenePage } from "./ScenePage";

registerPage("scene", ScenePage);
