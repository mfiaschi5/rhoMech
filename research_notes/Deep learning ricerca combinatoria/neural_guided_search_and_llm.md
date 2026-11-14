# Neural-guided tree search and LLM-driven search for combinatorial problems (state as of Nov 2026)

Scope note: web research done 2026-11-08. Facts verified in this session from primary or near-primary pages are marked with those links. Facts about well-known canonical papers (venue, year, headline number) that I did not re-fetch this session are cited to the canonical paper URL and flagged "(canonical, not re-fetched)". Secondary/aggregator claims are flagged as such.

## 1. AlphaZero/MuZero-style search beyond games (AlphaTensor, AlphaDev, Ranked Reward, AlphaProof/AlphaGeometry, MCTS for graph problems) and the critiques

### Takeaway
Policy/value-guided MCTS works convincingly when the problem can be cast as a single-player game with an exact, cheap verifier and an enormous but well-structured action space (tensor decompositions, assembly programs, formal proofs); DeepMind's results there are real and some are deployed. For classical graph CO (MIS, TSP), two careful re-evaluations (Böther et al. ICLR 2022; Xia et al. ICML 2024) showed that the learned network often contributes little beyond the search/algorithmic machinery, and classical solvers (KaMIS, LKH-3) remain as good or better.

### Cited Findings
**AlphaTensor / AlphaDev (game-ified algorithm discovery)**
- AlphaTensor (Fawzi et al., Nature 610, Oct 2022) casts matrix-multiplication algorithm discovery as a single-player "TensorGame" (low-rank tensor decomposition) solved by an AlphaZero-style agent with sampled MCTS; it found a 47-multiplication algorithm for 4x4 matrices in arithmetic mod 2 (vs. 49 by recursive Strassen) and improved the best known rank for many other small sizes — [Nature, AlphaTensor](https://www.nature.com/articles/s41586-022-05172-4) (canonical, not re-fetched).
- Follow-up showing search alternatives can compete: Kauers & Moosbauer's "flip graph" random-walk search improved some AlphaTensor mod-2 results shortly after (e.g., 5x5 mod 2) — [arXiv 2210.04045](https://arxiv.org/abs/2210.04045) (canonical, not re-fetched).
- AlphaDev (Mankowitz et al., Nature 618, June 2023) treats writing x86 assembly as a single-player "AssemblyGame" with AlphaZero-style search, rewarded by correctness and latency; discovered shorter sort3/sort5 routines merged into LLVM libc++ (DeepMind reported up to ~70% faster for short sequences, ~1.7% for sequences >250k elements) — [Nature, AlphaDev](https://www.nature.com/articles/s41586-023-06004-9); [DeepMind blog](https://deepmind.google/discover/blog/alphadev-discovers-faster-sorting-algorithms/) (canonical, not re-fetched).

**AlphaZero for CO**
- Ranked Reward (Laterre et al., 2018, arXiv 1807.01672) adapts AlphaZero self-play to single-player CO by rewarding the agent relative to a percentile of its own recent performance (a self-play surrogate), applied to 2D/3D bin packing — [arXiv 1807.01672](https://arxiv.org/abs/1807.01672) (canonical, not re-fetched).
- Abe et al. 2019, "Solving NP-hard problems on graphs with extended AlphaGo Zero" (CombOpt Zero; MVC, MaxCut, MIS) — [arXiv 1905.11623](https://arxiv.org/abs/1905.11623) (canonical, not re-fetched).
- Chemistry analogue (relevant for mechanism/route search): Segler, Preuss & Waller, Nature 555 (2018), 3N-MCTS retrosynthesis with neural expansion/rollout policies; Retro* (Chen et al., ICML 2020) uses A*-like best-first search with a learned value — [Nature 2018](https://www.nature.com/articles/nature25978); [Retro* arXiv 2006.15820](https://arxiv.org/abs/2006.15820) (canonical, not re-fetched).

**Formal math: AlphaProof / AlphaGeometry**
- AlphaProof + AlphaGeometry 2 scored 28/42 at IMO 2024 (silver-medal level); AlphaProof solved 3 of the 5 non-geometry problems including the hardest (P6) — [DeepMind blog, 25 Jul 2024](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/).
- AlphaProof paper: Hubert, Mehta, Sartran et al., "Olympiad-level formal mathematical reasoning with reinforcement learning", Nature, 12 Nov 2025 (DOI 10.1038/s41586-025-09833-y): AlphaZero-inspired agent searching for Lean proofs, trained by RL on millions of auto-formalized problems, with "Test-Time RL" (generating and training on problem variants at inference) for the hardest problems — [Nature Asia press release](https://www.natureasia.com/en/info/press-releases/detail/9147); [Google Research pub page](https://research.google/pubs/olympiad-level-formal-mathematical-reasoning-with-reinforcement-learning/).
- AlphaGeometry 2 (arXiv 2502.03544, v1 5 Feb 2025, v3 8 Dec 2025): solves 84% of all IMO geometry problems 2000–2024 vs 54% for AG1; formal-language coverage raised from 66% to 88%. (DeepMind's July 2024 blog quoted 83% over 25 years vs 53% — slightly different denominators/versions.) — [arXiv 2502.03544](https://arxiv.org/abs/2502.03544); [DeepMind blog](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/).
- AlphaGeometry 1 (Trinh et al., Nature, Jan 2024): neuro-symbolic — a language model proposes auxiliary constructions, a symbolic deduction engine searches; 25/30 on IMO-AG-30 — [Nature](https://www.nature.com/articles/s41586-023-06747-5) (canonical, not re-fetched).

**Critiques of DL-in-tree-search for CO**
- Böther, Kißig, Taraz, Cohen, Seidel, Friedrich, "What's Wrong with Deep Learning in Guided Tree Search for Combinatorial Optimization", ICLR 2022 (arXiv 2201.10494): re-implemented Li et al. (NeurIPS 2018) GCN-guided tree search for Maximum Independent Set; the GCN could be replaced by random values without hurting results — performance came from algorithmic components such as graph kernelization/reduction and local search; original results judged not reproducible; classical solvers (e.g., KaMIS) were often faster with similar quality; by contrast for the RL solver "Learning What To Defer" the GNN did drive solution quality. Released an open MIS benchmark suite (weighted/unweighted) — [arXiv 2201.10494](https://arxiv.org/abs/2201.10494); [ICLR 2022 poster](https://iclr.cc/virtual/2022/poster/6426); [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2201.10494).
- Xia et al., "Position: Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale TSP", ICML 2024 (PMLR 235; arXiv 2406.03503): in heatmap-guided MCTS (Att-GCN+MCTS, DIMES, UTSP, DIFUSCO line), a parameter-free-ish "SoftDist" heatmap (softmax of negative distances with one temperature tuned on the TSP objective) matches or beats learned heatmaps; heatmap-guided MCTS still trails LKH-3 on TSP-500/1000/10000 even with tuned MCTS and varied time budgets — [PMLR](https://proceedings.mlr.press/v235/xia24f.html); [code](https://github.com/xyfffff/rethink_mcts_for_tsp). A rebuttal comment by the UTSP author argues SoftDist timings omit GPU heatmap cost and that UTSP wins at equal budget on TSP-1000 — [comment summary](https://themoonlight.io/review/comment-on-paper-position-rethinking-post-hoc-search-based-neural-approaches-for-solving-large-scale-traveling-salesman-problems) (secondary).

### Inferences
- The successful "AlphaZero beyond games" cases share: a perfect, cheap verifier; a deterministic single-agent MDP; enormous search space where learned priors prune branching; and evaluation against the best known human results rather than against tuned OR solvers. In mainstream graph CO, where strong hand-engineered solvers exist, learned guidance is frequently a weak signal relative to reductions/local search — ablating the network (random/simple prior) is the essential control.
- Pattern relevant for ksmech: the Xia/Böther results are the CO analogue of "untrained physics kernel vs trained model" floors — a simple physically-motivated prior can be a strong baseline that learned priors must beat under equal compute.

### Gaps
- No primary-source numbers re-fetched for AlphaTensor, AlphaDev, Ranked Reward this session (cited from canonical memory). Post-AlphaDev discussion that some of its sort improvements were reproducible by simpler search/LLMs was not verified.
- No 2026 peer-reviewed systematic re-evaluation of neural MCTS for CO found beyond the 2022/2024 papers.

## 2. Learned heuristics for classical search (DeepCubeA, Neural A*, Levin tree search/PHS, beam search with learned scorers, learning-to-search, Monte Carlo beam search)

### Takeaway
Two families dominate: (a) learn a cost-to-go/value and plug it into (batched, weighted) A* — DeepCubeA is the flagship; (b) learn a policy and use it in Levin-style best-first search, which gives node-expansion bounds in terms of policy probability (LevinTS, PHS, LTS with context models). Learning-to-search (LaSO, Searn, DAgger, BSO) supplies the training theory for search-aware scorers; Monte Carlo beam/nested search provide strong learning-free baselines.

### Cited Findings
- DeepCubeA (Agostinelli, McAleer, Shmakov, Baldi, Nature Machine Intelligence 1:356–363, 2019): cost-to-go learned by deep approximate value iteration on states scrambled backward from the goal, used in batch weighted A*; solved 100% of Rubik's cube test configurations and found a shortest solution in ~60% of them; also 15/24/35/48-puzzles, Lights Out, Sokoban — [Nat. Mach. Intell.](https://www.nature.com/articles/s42256-019-0070-z) (canonical, not re-fetched).
- Neural A* (Yonetani et al., ICML 2021): differentiable A* trained end-to-end (encoder produces guidance cost maps) by imitating optimal paths for grid path planning — [arXiv 2009.07476](https://arxiv.org/abs/2009.07476) (canonical, not re-fetched).
- Levin Tree Search with a policy (Orseau, Lelis, Lattimore, Weber, NeurIPS 2018, "Single-agent policy tree search with guarantees"): best-first by cost d(n)/π(n); the number of expansions before reaching a goal is bounded by d/π of the goal; contrasted with PUCT/MCTS which lacks such guarantees — [arXiv 1811.10928](https://arxiv.org/abs/1811.10928) (canonical, not re-fetched).
- Policy-Guided Heuristic Search (PHS; Orseau & Lelis, AAAI 2021, pp. 12382–12390): combines LTS's policy with an A*-style heuristic, keeps LevinTS's upper bound and extends it to heuristics with a near-matching lower bound; on sliding-tile, Sokoban (Boxoban) and The Witness puzzles, learning policy+heuristic via a Bootstrap-style loop, PHS solved more problems in less time than A*, WA*, GBFS, LevinTS and PUCT, and was the only method good in all three domains — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17469); [arXiv 2103.11505](https://arxiv.org/abs/2103.11505); [UAlberta PDF](https://webdocs.cs.ualberta.ca/~santanad/papers/2021/orseauL21.pdf).
- LTS with context models (Orseau & Lelis 2023, arXiv 2305.16945): replaces the neural policy with online-convex "context models" so that the LTS loss is convex, giving convergence guarantees — [arXiv 2305.16945](https://arxiv.org/abs/2305.16945) (canonical, not re-fetched; venue believed IJCAI 2023, unverified).
- Subgoal search (kSubS; Czechowski et al., NeurIPS 2021): a learned generator proposes subgoals k steps ahead, searched by best-first with a learned value; strong on Sokoban, Rubik's cube, INT inequalities — [arXiv 2108.11204](https://arxiv.org/abs/2108.11204) (canonical, not re-fetched).
- Learning to search / imitation: LaSO (Daumé & Marcu, ICML 2005) trains a beam-search scorer with updates when the gold prefix falls off the beam; Searn (Daumé, Langford, Marcu, MLJ 2009); DAgger (Ross, Gordon, Bagnell, AISTATS 2011) gives no-regret imitation by training on the learner's own state distribution; Beam Search Optimization (Wiseman & Rush, EMNLP 2016) brings LaSO-style losses to seq2seq — [DAgger arXiv 1011.0686](https://arxiv.org/abs/1011.0686); [BSO arXiv 1606.02960](https://arxiv.org/abs/1606.02960) (canonical, not re-fetched).
- Learned scorers inside exact solvers: Gasse et al. (NeurIPS 2019) imitate strong branching with a GCNN for branch-and-bound in SCIP — [arXiv 1906.01629](https://arxiv.org/abs/1906.01629) (canonical, not re-fetched).
- Monte Carlo search family (learning-free strong baselines): Nested Monte Carlo Search (Cazenave, IJCAI 2009), Monte Carlo Beam Search (Cazenave, IEEE CIG 2012), Nested Rollout Policy Adaptation (Rosin, IJCAI 2011; Morpion Solitaire records) — (canonical, not re-fetched; no URL verified this session).

### Inferences
- Policy-guided best-first (LTS/PHS) is the natural fit when you have a calibrated next-step distribution and want expansion bounds; value-guided A* fits when a good cost-to-go can be learned by backward generation from goals (DeepCubeA trick). For an autoregressive step predictor (e.g., mechanism steps), LTS-style cost d/π over step sequences is a principled alternative to plain beam search.
- LaSO/DAgger matter because a scorer trained only on gold prefixes is miscalibrated on the states beam search actually visits (exposure bias); search-aware training is the standard fix.

### Gaps
- Exact PHS/LTS per-domain numbers were not retrieved (tables in the PDF).
- No 2025–2026 survey on learned heuristics for classical search verified this session.

## 3. LLM-driven search (FunSearch, AlphaEvolve, EoH, ReEvo, LLaMEA, OpenELM, open-source clones, tree-of-thoughts/test-time compute): results and limitations through 2026

### Takeaway
LLM-as-mutation-operator inside an evolutionary loop with an automatic evaluator (FunSearch -> AlphaEvolve and open clones) is now the dominant "LLM search" paradigm for constructive math and heuristic design; it has produced verified new bounds (cap sets, 4x4 complex matrix multiplication with 48 multiplications, kissing number in 11D, dozens of analysis/combinatorics constructions) and deployed systems (Borg scheduling). Limitations: dependence on a cheap, exploitable evaluator; fragile generated code; weak evaluation practice (single seed/budget); gains on realistic constrained OR problems are small or unproven; priority claims sometimes complicated by prior literature.

### Cited Findings
**FunSearch (Romera-Paredes et al., Nature 625, published Dec 2023)**
- Pairs a pretrained code LLM with an automatic evaluator, evolving programs (a "priority" function inside a fixed skeleton) via an island-based database; found a cap set of size 512 in dimension 8 (previous best 496) and improved lower bounds on cap-set capacity via admissible sets; evolved online bin-packing heuristics beating first-fit/best-fit on OR-Library and Weibull benchmarks — [Nature](https://www.nature.com/articles/s41586-023-06924-6); [Wikipedia summary](https://en.wikipedia.org/wiki/FunSearch). The size-512 cap set reportedly appeared in only 4 of 140 runs — [Vinitsky blog](https://eugenevinitsky.github.io/posts/funsearch) (secondary).

**AlphaEvolve (DeepMind, 14 May 2025; white paper arXiv 2506.13131, Novikov et al.)**
- Method: Gemini Flash (breadth) + Gemini Pro (depth) propose whole-program diffs; automated evaluators score programs; an evolutionary database feeds strong programs back into prompts — [DeepMind blog](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/).
- Results (DeepMind blog): Borg data-center scheduling heuristic recovering on average 0.7% of Google's worldwide compute, in production >1 year; a Gemini matmul kernel 23% faster -> 1% reduction in Gemini training time; up to 32.5% FlashAttention kernel speedup; Verilog simplification of a TPU matmul circuit; 4x4 complex-valued matrix multiplication with 48 scalar multiplications (beating Strassen's 49 for this setting); on >50 open math problems ~75% matched SOTA and ~20% improved it; kissing number in 11 dimensions raised to a 593-sphere configuration — [DeepMind blog](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/).
- Priority nuance: Dumas, Pernet & Sedoglavic (June 2025, arXiv 2506.13242) derived a 48-multiplication 4x4 algorithm with rational coefficients from AlphaEvolve's complex one (valid over rings with 1/2); a March 2026 note gives a more numerically accurate variant (error-bound exponent ≈2.386 vs ≈2.577) — [arXiv 2506.13242](https://arxiv.org/abs/2506.13242v4); [arXiv 2603.18699](https://arxiv.org/abs/2603.18699v1); also [arXiv 2602.13171](https://arxiv.org/html/2602.13171v1) generalizes complex-to-rational conversion.
- Georgiev, Gómez-Serrano, Tao, Wagner, "Mathematical exploration and discovery at scale" (arXiv 2511.02864, Nov 2025): AlphaEvolve on 67 problems in analysis, combinatorics, geometry, number theory; rediscovered best known constructions in most cases and improved several (e.g., finite-field Kakeya bounds); problem repository released on GitHub (google-deepmind/alphaevolve_repository_of_problems) — [arXiv 2511.02864](https://arxiv.org/abs/2511.02864); [Tao blog, 5 Nov 2025](https://terrytao.wordpress.com/2025/11/05/mathematical-exploration-and-discovery-at-scale/).
- A May 2026 DeepMind follow-up post/paper on scaling AlphaEvolve's impact across fields is reported by secondary sources (claims include DNA-sequencing error reductions); AlphaEvolve itself is available only via a Google Cloud private-preview waitlist — [secondary blog](https://aitechconnect-976.pages.dev/news/alphaevolve-deepmind-gemini-algorithm-discovery) (unverified, aggregator). I found no 2026 Nature paper on AlphaEvolve.

**Open-source successors (2025–2026)**
- ShinkaEvolve (Lange, Imajuku, Cetin — Sakana AI; arXiv 2509.19349, Sep 2025): sample-efficient program evolution; new SOTA circle-packing solution with only 150 samples; also AIME agent harnesses, ALE-Bench competitive-programming improvements (~2.3% average over ALE-Agent across 10 tasks), MoE load-balancing losses — [arXiv 2509.19349](https://arxiv.org/abs/2509.19349).
- CodeEvolve (arXiv 2510.14150; Findings of EMNLP 2026): matches or surpasses AlphaEvolve's *reported* results on 5 of 9 problems and beats OpenEvolve and ShinkaEvolve under matched conditions on 6 of 9 — [arXiv](https://arxiv.org/html/2510.14150v4); [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1345/).
- Low-cost study: LLM-guided program evolution broke 10 Packomania circle-packing records for about $28 on a consumer PC (arXiv 2609.05093, Sep 2026) — [arXiv](https://arxiv.org/html/2609.05093v1).
- FrontierOR (arXiv 2605.25246, May 2026): benchmark of LLMs' capacity for efficient algorithm design in large-scale optimization — [arXiv](https://arxiv.org/pdf/2605.25246) (abstract details not retrieved).
- A Towards-AI blog reports OpenEvolve "cheating" (reward hacking the evaluator) across 18 algorithms — [blog](https://pub.towardsai.net/i-pointed-openevolve-at-18-algorithms-the-6-2k-star-alphaevolve-clone-cheated-by-321-8912d8f11e72) (secondary, unverified; illustrates evaluator exploitation).

**LLM automated heuristic design (AHD) line**
- OpenELM / Evolution through Large Models (Lehman et al., 2022): LLMs as mutation operators in MAP-Elites (Sodarace) — [arXiv 2206.08896](https://arxiv.org/abs/2206.08896) (canonical, not re-fetched).
- EoH, Evolution of Heuristics (Liu et al., ICML 2024): co-evolves natural-language "thoughts" and code — [arXiv 2401.02051](https://arxiv.org/abs/2401.02051); ReEvo (Ye et al., NeurIPS 2024): reflective evolution with short- and long-term verbal reflections, applied to TSP/CVRP/BPP and to ACO/GLS heuristics — [arXiv 2402.01145](https://arxiv.org/abs/2402.01145); LLaMEA (van Stein & Bäck, IEEE TEVC): LLM-generated metaheuristics for continuous black-box optimization (BBOB) — [arXiv 2405.20132](https://arxiv.org/abs/2405.20132) (all canonical, not re-fetched).
- Zhang, Liu, Lin, Wang, Lu, Zhang, "Understanding the Importance of Evolutionary Search in Automated Heuristic Design with LLMs", PPSN 2024: prior AHD work had inconsistent benchmarks, weak baselines, little component analysis; benchmark of 4 LLM-EPS methods x 4 problems x 9 LLMs x 5 runs supports that evolutionary search (not just the LLM) matters — [arXiv 2407.10873](https://arxiv.org/abs/2407.10873).
- HSEvo (AAAI 2025): EoH shows higher diversity than FunSearch and ReEvo but unstable objective; ReEvo's reflection gives good scores but poor diversity — [arXiv 2412.14995](https://arxiv.org/html/2412.14995v1).
- MeLA (arXiv 2507.20541): EoH and ReEvo frequently generate non-executable heuristics; ReEvo's crossover stalls when no executable parents exist; both struggle on real-world problems — [arXiv](https://arxiv.org/pdf/2507.20541).
- 3D packing re-evaluation (arXiv 2509.02297): direct EoH application infeasible due to code fragility; as constraints (load stability, item separation) are added, LLM-discovered heuristics lose effectiveness; the scoring function alone cannot navigate real-world trade-offs; pretrained biases a barrier — [review](https://www.themoonlight.io/review/re-evaluating-llm-based-heuristic-search-a-case-study-on-the-3d-packing-problem); [emergentmind](https://www.emergentmind.com/papers/2509.02297) (secondary summaries).
- Oved, Pony, Naparstek, Barzelay, "Evolution or Illusion? Rethinking Evaluation in LLM Evolutionary Search" (arXiv 2609.19799, 17 Sep 2026): single-seed, fixed-iteration reporting is insufficient; the best split of budget between seeds (width) and iterations (depth) depends on strategy/task/budget; method rankings flip with budget (worst at 1 seed can be best at 40 seeds); common iteration counts waste budget better spent on more seeds; proposes reporting the seeds-by-iterations frontier — [arXiv](https://arxiv.org/abs/2609.19799).

**LLM test-time search as combinatorial search**
- Tree of Thoughts (Yao et al., NeurIPS 2023): BFS/DFS over LLM "thoughts" with LLM self-evaluation; Game of 24 success 4% (GPT-4 CoT) -> 74% (ToT) — [arXiv 2305.10601](https://arxiv.org/abs/2305.10601) (canonical, not re-fetched).
- Snell et al. 2024, "Scaling LLM test-time compute optimally": a compute-optimal allocation (verifier-guided search vs revisions, by difficulty) improves efficiency >4x over best-of-N — [arXiv 2408.03314](https://arxiv.org/abs/2408.03314) (canonical, not re-fetched).

### Inferences
- The decisive ingredient across FunSearch/AlphaEvolve is the evaluator: programs are cheap to score exactly, and evolution optimizes against it. Where evaluation is noisy, slow or gameable (real-world OR with side constraints), gains shrink and reward hacking appears.
- Searching in program space (heuristics/constructions) rather than solution space gives interpretability and transfer across instances — the reason bin-packing heuristics and Borg scheduling deploy.
- 2026 evaluation literature suggests reported differences between LLM-evolution frameworks are often within seed/budget variance; comparisons should report multiple seeds and compute frontiers.

### Gaps
- No primary DeepMind source verified for the May 2026 AlphaEvolve impact update.
- Exact FunSearch bin-packing excess-bin percentages and admissible-set capacity bound not re-fetched.
- No independent 2026 benchmark comparing AlphaEvolve itself (closed) against open frameworks under identical conditions was found.
- Whether random/naive LLM sampling baselines match evolutionary loops on standard AHD tasks remains unsettled in the sources retrieved.

## 4. Neural program synthesis and neural-guided deductive search (DeepCoder, DreamCoder, NGDS)

### Takeaway
The pre-LLM template for "neural guidance + symbolic search": a network predicts which primitives/rules are likely, and an enumerative or deductive search uses those predictions for ordering/pruning, with the symbolic engine guaranteeing correctness. DreamCoder adds library learning so the search space itself is reshaped over time.

### Cited Findings
- DeepCoder (Balog et al., ICLR 2017): a network predicts DSL-function presence from I/O examples; used to order DFS/sort-and-add enumeration, giving roughly an order-of-magnitude speedup over unguided search — [arXiv 1611.01989](https://arxiv.org/abs/1611.01989) (canonical, not re-fetched).
- Neural-Guided Deductive Search (Kalyan et al., ICLR 2018): neural model scores branches of PROSE/FlashFill-style deductive synthesis for real-time synthesis from few examples, reported speedups up to ~12x — [arXiv 1804.01186](https://arxiv.org/abs/1804.01186) (canonical, not re-fetched).
- DreamCoder (Ellis et al., PLDI 2021): wake-sleep loop alternating search for programs guided by a neural recognition model, abstraction (library learning by refactoring) and "dreaming" (training the recognition model on replays/fantasies) — [arXiv 2006.08381](https://arxiv.org/abs/2006.08381) (canonical, not re-fetched).

### Inferences
- FunSearch/AlphaEvolve are the LLM-era continuation: the LLM replaces the recognition/policy model, the evaluator replaces the specification check, and the evolutionary database plays the role of a (soft) library.

### Gaps
- No 2025–2026 primary sources retrieved on neural-guided deductive search specifically.

## 5. What makes guidance effective: value vs policy, exploration, compute trade-offs, generalization

### Takeaway
Guidance helps most when (i) the verifier is exact and cheap, (ii) the learned signal is evaluated against strong non-learned baselines under equal compute, and (iii) the search algorithm exploits the signal's form — policies in Levin-style/PUCT search (bounds via π), values in A*/best-first (DeepCubeA), both combined in PHS/AlphaZero. Exploration/diversity mechanisms (islands, MCTS exploration constants, multiple seeds) are as important as the model's quality; generalization beyond training size is the recurrent weak point of learned CO guidance.

### Cited Findings
- Policy + heuristic combined (PHS) outperformed pure-policy (LevinTS, PUCT) and pure-heuristic (A*, WA*, GBFS) on all three domains, with every single-signal method failing badly in at least one — [UAlberta PDF](https://webdocs.cs.ualberta.ca/~santanad/papers/2021/orseauL21.pdf); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17469).
- Random or simple priors can match learned priors inside strong search: random GCN outputs in MIS tree search (Böther 2022) and SoftDist heatmaps in TSP MCTS (Xia 2024) — [arXiv 2201.10494](https://arxiv.org/abs/2201.10494); [PMLR](https://proceedings.mlr.press/v235/xia24f.html).
- In LLM evolution, diversity vs exploitation trade-offs differ by method (EoH diverse but unstable; ReEvo exploitative) — [HSEvo](https://arxiv.org/html/2412.14995v1); seeds-vs-iterations budget split changes rankings — [arXiv 2609.19799](https://arxiv.org/abs/2609.19799); evolutionary search component shown to matter beyond the LLM — [arXiv 2407.10873](https://arxiv.org/abs/2407.10873).
- AlphaEvolve's use of a fast model for breadth and a strong model for depth is an explicit compute/quality trade-off — [DeepMind blog](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/); ShinkaEvolve emphasizes sample efficiency (SOTA circle packing in 150 samples) — [arXiv 2509.19349](https://arxiv.org/abs/2509.19349).
- Test-time RL (AlphaProof) — adapting the policy on problem variants at inference — was needed for the hardest problems, showing that static policies may not generalize to out-of-distribution hard instances — [Google Research](https://research.google/people/106726/?type=google).
- Classical solvers retain an edge in well-studied CO: KaMIS-like solvers often faster at similar quality for MIS; LKH-3 ahead of heatmap+MCTS for large TSP — [arXiv 2201.10494](https://arxiv.org/abs/2201.10494); [PMLR](https://proceedings.mlr.press/v235/xia24f.html).

### Inferences
- Practical checklist distilled from the sources: (1) always ablate the learned signal (random/uniform/simple physics prior) inside the same search; (2) compare at equal wall-clock and hardware, counting network inference; (3) report several seeds and a compute frontier, not one budget; (4) prefer search algorithms whose guarantees match the signal (policy -> LTS/PUCT, value -> A*); (5) train scorers on states the search actually visits (LaSO/DAgger) to avoid exposure bias; (6) expect generalization failures on larger/harder instances and test them explicitly.
- For a stepwise mechanism predictor, these map to: beam search vs Levin-style best-first over elementary steps; a physics-only floor ranking as the "SoftDist" control; and search-aware training.

### Gaps
- No quantitative cross-paper meta-analysis of value- vs policy-guidance found.
- Gumbel MuZero (policy improvement with few simulations; Danihelka et al., ICLR 2022) and MuZero itself (Schrittwieser et al., Nature 2020) are relevant to low-simulation regimes but were not re-verified this session.
