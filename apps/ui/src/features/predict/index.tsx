// The Predict page (design §11.1 feature seam): registered for the shell, which imports this
// file eagerly. This feature never imports another one; it talks to the Curate page only through
// hash links (#/curate/<jobId>?pathway=<pid>). Its stylesheet also styles the Correct page.
import { registerPage } from "../../registry";
import { PredictPage } from "./PredictPage";
import "./predict.css";

registerPage("predict", PredictPage);
