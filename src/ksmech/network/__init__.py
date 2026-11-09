"""The reaction network in the engine (reaction-network plan, items C, D, I1, I2, J).

A network is an implicit graph that the models generate on demand: nodes are pool states,
merged by pool key (the multiset of species identity keys, so up to resonance), and edges are
elementary steps with the predictor's log P. Expanding a node costs seconds (QM for new
molecules, tokens, the model), so every algorithm here is measured by the nodes it expands.

- `keys`: the pool key (shared with the web app's network);
- `graph`: pure algorithms over an explored graph: best-first simple paths and the turnover
  rule of a cycle (moved out of `serve.predict.network`, which imports them);
- `search`: search over the implicit graph: the `Expander` protocol, the expansion cache, the
  k best pathways to a goal by best-first search with an admissible heuristic (A* for k = 1),
  and the mechanism classes up to the order of independent steps;
- `target`: the arrival test on a target product (exact identity key, or the heavy-atom
  skeleton up to protons) and the remaining-change heuristic from the atom map;
- `pools`: the engine's pool states (Kekule, explicit-H, every atom mapped) and the model's
  expander over them (`model.decode`).

The species graph (item J) keeps branches that join apart: OR nodes are species, AND nodes
elementary steps with their atom maps.

- `species`: the graph and its expansion species by species without a product, partners at
  fixed concentration;
- `lift`: between pool states and species (canonical local atoms, pools built from species,
  pool steps split into species steps, a pool-graph pathway compared with a hyperpath);
- `hyperpaths`: SBT, the k shortest hyperpaths, and their linearization into pool-graph
  pathways;
- `flux`: masses, step fluxes and atom-flux analysis (the fixed-concentration approximation).

The engine never imports `ksmech.serve`; the web app builds its own network on `graph`.
"""
