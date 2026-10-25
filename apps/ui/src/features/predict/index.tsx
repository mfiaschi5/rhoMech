// The Predict and Correct pages (design §11.1 feature seam): registered for the shell, which
// imports this file eagerly. This feature never imports another one; it talks to the Curate
// page only through hash links (#/curate/<jobId>?pathway=<pid>).
import { registerPage } from "../../registry";
import { CorrectPage } from "./CorrectPage";
import { PredictPage } from "./PredictPage";
import "./predict.css";

registerPage("predict", PredictPage);
registerPage("correct", CorrectPage);
