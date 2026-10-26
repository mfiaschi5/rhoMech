// The embedding endpoints over api/client.ts's request(): the local checkpoints and the
// trace job. Shapes: types.gen.ts (CheckpointOut, ModelsOut, EmbeddingIn, EmbeddingResultOut).
import { request, type RequestOptions } from "../../api/client";
import type { EmbeddingIn, JobCreatedOut, ModelsOut } from "../../api/types.gen";

/** GET /models: the local checkpoints and the embedding capability they give. */
export const getModels = (o?: RequestOptions) => request<ModelsOut>({ path: "/models" }, o);

/** POST /embedding: a job (kind "embedding", result EmbeddingResultOut) that traces a pool
 *  state or a SMILES through a checkpoint. */
export const createEmbedding = (body: EmbeddingIn, o?: RequestOptions) =>
  request<JobCreatedOut>({ method: "POST", path: "/embedding", body }, o);
