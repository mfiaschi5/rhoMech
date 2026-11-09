"""The reaction network in the engine (reaction-network plan, items C, D, E, F, I1-I4).

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
  expander over them (`model.decode`);
- `free_energy`: the free energies the chain reads (G, H, S of a state, mu of a reservoir
  species; `load_thermo_model` is the adapter point of item B), K(T), delta(T) and T* = dH/dS;
- `reservoir`: species always available (solvent, reagents in excess), kept in every pool, their
  draws priced at their chemical potential;
- `chain`: the explored graph as a Markov chain: reverse edges scored at the children, detailed
  balance with the free energies (Boltzmann stationary), absorption probabilities;
- `stop`: the thermodynamic stop: SINK_AT_T after a rescan with more candidates, NO_NET_REACTION;
- `tpt`: transition path theory: committors, reactive flux, dominant pathways by bottleneck;
- `explore`: what forms without a product, best-first by probability mass, with the unexplored
  mass as the uncertainty.

The engine never imports `ksmech.serve`; the web app builds its own network on `graph`.
"""
