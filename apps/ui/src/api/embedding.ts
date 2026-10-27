// The embedding router (serve/embedding): the trace job. The local checkpoints it runs on are
// GET /models (predict.ts). Shapes: types.gen.ts (EmbeddingIn, EmbeddingResultOut).
import { request, type RequestOptions } from "./http";
import type { EmbeddingIn, JobCreatedOut } from "./types.gen";

/** POST /embedding: a job (kind "embedding", result EmbeddingResultOut) that traces a pool
 *  state or a SMILES through a checkpoint. */
export const createEmbedding = (body: EmbeddingIn, o?: RequestOptions) =>
  request<JobCreatedOut>({ method: "POST", path: "/embedding", body }, o);
