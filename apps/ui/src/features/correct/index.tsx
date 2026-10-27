// Feature seam (design §11.1): registers the "correct" page. It reuses the Predict feature's
// engine modules (pool canvas, R matrix, step views, the v9 hooks) and its stylesheet, never
// its page.
import { registerPage } from "../../registry";
import { CorrectPage } from "./CorrectPage";

registerPage("correct", CorrectPage);
