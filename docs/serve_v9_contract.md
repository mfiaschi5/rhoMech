# ksmech-serve: the spec-v9 Predict / Correct API contract

## Changes against the brief (read first)

Every name of the brief is kept. These are additions or clarifications:

1. **Pool atoms and the depiction.** `DepictAtomOut` gains `pool_atom: number` (the atom
   index of `parse_mapped(state.mapped_smiles)`, explicit H included, the index space of every
   `MatrixIO`, `ArrowReadingOut` and `OrbitalReadingOut`) and `hidden_h_atoms: number[]` (the
   pool atoms of the H atoms the depiction hides on that atom). `depiction_idx` is the molblock
   order and is **not** a pool atom; `global_site` is the token index space and is **not** a
   pool atom either. Pass `keep_maps` (map numbers) to draw chosen H atoms explicitly.
   Do not derive pool atoms from map numbers: a state's mapped SMILES is canonicalized, so
   `map = pool atom + 1` holds for a root built from the input but not after a step (a
   product keeps its parent's maps in a new atom order). Use `pool_atom`.
2. **`OrbitalReadingOut`** also has `term` (the channel's term of G_atom, 1/eV). `canonical`
   also has `k_label`, `m_label` ("HOMO-1", "LUMO"), `term`, `share` and `index_source`. `i` is
   the donor end of the forming bond in the arrow reading (when the reading has one), `j` the
   other end. A forming bond without any valid donor/acceptor channel has no reading.
3. **`EdgeOut`** also has `weight` (log beam weight, 0 for a user step), `pmechdb_code` (the
   reading as a PMechDB arrow code on map numbers), `formal_charges`, `touched_maps`,
   `channel` (diagnostic: how the predictor completed R), `alternatives` (other R with the
   same product: `{matrix, anchor, score, logp, rank}`). `score` is `number | null` and
   `anchor` is `AnchorOut | null`. A user step has `kind: "corrected"`, `predictor: "user"`,
   `rank: 0`, `logp: null` and `score` = |G_atom| of its anchor (null without readings).
   Removed v3 fields: `arrows: ArrowOut[]` (now `ArrowReadingOut[]`), `dataset_arrows`,
   `encoding_base_smiles`, `encode_reason`, `probability`, `original`, `predictor_view`.
4. **`StepCheckOut`** also has `anchor`, `product_state_id` (the product state, saved, so
   `/api/pool/{id}/depiction` works on it), `pmechdb_code` and `readings_status`:
   `"ready"` | `"pending"` (the pool's analysis is being computed: re-check later, the
   arrows are read without the gap meanwhile) | `"unavailable"` (the pool has no tokens).
5. **Override is synchronous** and answers `OverrideResultOut {edge, node}` (200). It may
   answer **202 `PendingOut`** while the parent state's analysis is computed (retry), **422**
   `STEP_INVALID` with `error.detail.issues` (the `StepIssueOut` list) when R is no valid
   step, **409** `NETWORK_BUSY` while a job runs on the network. The same R from the same
   state is the same edge (idempotent).
6. **Expansion of a node** is a job: `POST /api/jobs/{job_id}/expand {node_id, steps?: 1,
   predictor?: string, beam?: number}` -> 202 `JobCreatedOut` (job kind `"expand"`, result
   `ExpandResultOut {network_id, network_revision, node_id, predictor, n_nodes, n_edges}`).
   The default predictor is the network's (`"floor"` for a user network).
7. **Correct without a prediction**: `POST /api/predict` with `predictor: "user"` creates a
   job whose network is the root alone (status done, nothing expanded). Add steps with
   `override` and expand nodes with `expand`. `POST /api/pools` still only builds and saves a
   pool state (no job): use it with `POST /api/step/check` for checks before a job exists.
8. **Predictors**: `PredictorInfoOut` keeps its other fields (`label`, `code`, `version`,
   `checkpoint_hash`, `score_unit`, `default_temperature`, `warnings`, `description`,
   `spec_version`). `kind` equals `name`.
9. **Pathways**: `PathwayOut.probability` is `null` unless the predictor gives probabilities;
   `logp` is then the summed beam weights. `NetworkOut.scores_are_probabilities` says which.
10. **Export** is `GET /api/curate/export/pmechdb?status=validated&status=corrected` (a CSV
    download in PMechDB's raw format; report in the `X-Ksmech-Export-Report` header). The
    old `POST /api/curate/export` job and `GET /api/curate/export/latest` are removed.
11. **Removed** routes: `/api/pool/{id}/candidates`, `/api/pool/{id}/physics`,
    `/api/pool/{id}/site_pairs`, `/api/contact/...` (and their schemas). The capabilities
    `candidates` and `physics` stay listed, off, with the reason; `site_pairs` and `contact`
    are gone.
12. `FailedProposalOut` is `{node, rank, matrix, score, code, detail}`.

## Index spaces

- **Pool atoms**: `parse_mapped(state.mapped_smiles)` atom indices; every H is an atom. All
  matrices, arrows and readings use them, on the *source* state of the step.
- **Tokens**: global token ids of `GET /api/pool/{id}/tokens` (atoms first, then orbitals).
  `donor_token` / `acceptor_token` are orbital token ids.
- Map numbers appear only in `pmechdb_code`, `touched_maps` and the curated StepDoc's
  `dataset_arrows`.

## Types

```ts
MatrixIO = { bonds: [i, j, delta][]; lone: [i, delta][] }   // i < j; bond delta ±1 (±2/±3
                                                              // allowed); lone delta ±2 (±4/±6)
AnchorOut = { i: number; j: number; sign: "form" | "break" }
ArrowReadingOut = {
  kind: "lone_pair_to_bond" | "bond_to_bond" | "bond_to_lone_pair";
  source: number[];  // [a] lone pair on a, or [a, c] the bond whose pair moves
                     // (bond_to_bond: a is the shared atom, the new bond forms at a;
                     //  bond_to_lone_pair: the receiving atom is the last one)
  target: number[];  // [a, b] the forming bond (a the donor end), or [k] the receiving atom
}
OrbitalReadingOut = {
  i: number; j: number; g_atom: number; donor_token: number; acceptor_token: number;
  g_rs: number; term: number; donor_diag: string | null; acceptor_diag: string | null;
  canonical: { k: number; m: number; e_k_ev: number; e_m_ev: number; k_label: string;
               m_label: string; term: number; share: number | null; index_source: string } | null
}
StepIssueOut = { code: string; message: string; atoms: number[] }
```

Issue codes: `EMPTY_MATRIX`, `MATRIX_MALFORMED` (an index outside the pool, i = j, a delta
not allowed), `CHARGE_NOT_CONSERVED`, `BOND_ORDER_RANGE` (an order would leave 0..3),
`BREAK_UNBONDED` (a negative change on a pair without a bond), `LONE_PAIR_NEGATIVE`,
`NO_LONE_PAIR` (a -2 on an atom without a lone pair), `VALENCE_EXCEEDED` (RDKit's valence
check, or more than an octet / duet on H-Ne), `SANITIZE_FAILED`, `RESONANCE_ONLY_STEP` (the
product has the parent's pool key), `NO_ARROW_READING` (R does not decompose into
electron-pair moves).

## Endpoints

| Method | Path | Body | Answer |
|---|---|---|---|
| GET | `/api/predictors` | | `PredictorInfoOut[]` `{name, kind: "floor"\|"gbdt"\|"network", available, reason, version, scores_are_probabilities, ...}` |
| POST | `/api/predict` | `PredictIn` (`reactants_smiles`, `stoichiometry`, `predictor` = floor\|gbdt\|network\|user, `beam` = 5, `max_steps` = 8, ...) | 202 `JobCreatedOut` |
| GET | `/api/jobs/{job_id}` | | `JobOut` (`result`: `NetworkOut` with `EdgeOut[]`) |
| POST | `/api/step/check` | `{state_id, matrix: MatrixIO}` | `StepCheckOut {ok, issues, product_smiles, sigma_keys, formal_charges, arrows, readings, anchor, product_state_id, pmechdb_code, readings_status}` |
| POST | `/api/jobs/{job_id}/override` | `{parent_state_id, matrix: MatrixIO}` | `OverrideResultOut {edge: EdgeOut, node: NodeOut}` |
| POST | `/api/jobs/{job_id}/expand` | `{node_id, steps?, predictor?, beam?}` | 202 `JobCreatedOut` |
| POST | `/api/pools` | `{reactants_smiles, stoichiometry?}` | `PoolOut` (a saved state, no job) |
| GET | `/api/pool/{state_id}/depiction` | `?keep_maps=1,7` | `DepictionOut` (atoms with `pool_atom`, `hidden_h_atoms`) |
| POST | `/api/curate` | `CurateIn {job_id, pathway_id \| edge_ids, action}` | 201 `MechanismOut` |
| POST | `/api/curate/{mid}/recheck` | `{predictor?}` | 202 `JobCreatedOut` (rows: `first_arrow_rank` = rank of the curated product among the predictor's distinct products, compared by sigma keys) |
| GET | `/api/curate/export/pmechdb` | `?status=` | `text/csv` download |

`EdgeOut` keeps `id, source, target, target_pool_key, kind, status, source_state_id,
product_state_id, source_mapped_smiles, product_mapped_smiles, predictor_version,
n_alternatives, superseded_by, recurrence, closes_cycle, cycle_ids, revision` and has
`matrix: MatrixIO, anchor, arrows: ArrowReadingOut[], readings: OrbitalReadingOut[],
score, logp: number | null, predictor, rank, product_smiles: string[], sigma_keys: string[]`
(product fragment order) plus the additions of item 3.

Scores are not probabilities for `floor` (|G_atom| of the anchor, 1/eV) and `gbdt` (a
lambdarank score); `scores_are_probabilities` is true only for `network`.
