# Machine learning inside exact combinatorial solvers (MIP branch-and-bound, SAT, CP): state as of late 2026

Scope note: web research only (Nov 2026). Numbers are the authors' own unless stated otherwise. Items I could not check against a primary or near-primary source are listed under Gaps. "Survey" below means Scavuzzo, Aardal, Lodi, Yorke-Smith, "Machine learning augmented branch and bound for MILP", Mathematical Programming 2024 (arXiv 2402.05501). I read its full text.

## 1. Learned branching and node selection: what is reported, and on which benchmarks?

### Takeaway
Gasse et al. (NeurIPS 2019) is the turning point. A bipartite-graph GCNN trained to imitate strong branching beat SCIP's default reliability pseudocost branching (RPB) on synthetic instance families: up to about 1.4x faster in shifted geometric mean time on medium set covering and combinatorial auctions, and roughly on par on hard maximum independent set and facility location. Every follow-up keeps one of three framings: imitation of (full) strong branching (Gupta's hybrid models, DeepMind's Neural Branching), tree-context generalization (Zarpellon), or RL from scratch (Etheve; Scavuzzo's tree MDPs). Gains are consistent on homogeneous, generator-based families. On heterogeneous sets, reliability branching often still wins. Node selection shows only small margins.

### Cited Findings
**Early work**
- He, Daumé & Eisner (2014) learn node selection by imitating an oracle that always expands the node on the path to the optimal solution. The optimal solution is needed at training time only. — [Survey, arXiv 2402.05501](https://arxiv.org/abs/2402.05501v1)
- Khalil et al. (2016) and Alvarez et al. learn branching online, per instance: a strong-branching initialization at the top of the tree, then a linear model. Khalil frames the task as ranking rather than score regression. Feature-weight analysis found only weak correlation between the models learned on different instances. — [Survey](https://arxiv.org/abs/2402.05501v1)

**Gasse et al. 2019 (GCNN imitation of strong branching)**
- Setup: SCIP 6.0.1, 1 h limit, trained on "easy" instances only.
  - Set covering: easy 500 rows x 1000 cols, medium 1000 rows, hard 2000 rows.
  - Combinatorial auction: easy 100 items / 500 bids, medium 200/1000, hard 300/1500.
  - Capacitated facility location: easy 100 customers, medium 200, hard 400.
  - Maximum independent set: easy 500 nodes, medium 1000, hard 1500.

  — [Gasse et al. 2019, ar5iv 1906.01629](https://ar5iv.labs.arxiv.org/html/1906.01629)
- Shifted geometric mean time in seconds, GCNN vs RPB (SCIP default):

  | Benchmark | Easy | Medium | Hard |
  |---|---|---|---|
  | Set covering | 6.59 vs 8.98 | 42.48 vs 60.07 (GCNN won 100/100) | 1489.91 vs 1677.02 (66/70 wins; nodes 29,981 vs 47,299) |
  | Combinatorial auction | 1.85 vs 2.74 | 10.29 vs 17.41 | 114.16 vs 136.17 |
  | Facility location | 22.10 vs 26.55 | 120.94 vs 156.12 | 563.36 vs 631.50 |
  | Max independent set | 6.43 vs 8.77 | 192.91 vs 110.99 (RPB better) | 2024.37 vs 2045.61 |

  Full strong branching (FSB) timed out on hard set covering and hard MIS. The GCNN also beat the earlier learned rankers (SVMrank, LambdaMART, ExtraTrees). — [Gasse et al. 2019](https://ar5iv.labs.arxiv.org/html/1906.01629)
- The survey calls Gasse's result "a breakthrough in the learning to branch literature": the first learned rule to outperform reliability branching. Gupta et al. later modified the loss to improve it further, and Seyfi et al. added the history of past branchings. — [Survey](https://arxiv.org/abs/2402.05501v1)

**Generalization and the tree context**
- Zarpellon et al. (AAAI 2021) add a representation of the search tree's state. Gasse's model "struggles to learn over a heterogeneous data distribution", and tree context helps generalization. Even so, "reliability branching still outperforms both Zarpellon et al. and Lin et al.", possibly because of the cost of computing the descriptors. — [Survey](https://arxiv.org/abs/2402.05501v1)

**DeepMind: Neural Diving and Neural Branching**
- Nair et al. (arXiv 2012.13349, Dec 2020, revised Jul 2021; 19 DeepMind/Google authors) add two components to SCIP:
  - Neural Diving predicts partial assignments, and SCIP solves the resulting sub-MIPs.
  - Neural Branching imitates a GPU-scalable variant of full strong branching (ADMM-based).
- They evaluate on six datasets, including two Google production sets and MIPLIB, with 10^3 to 10^6 variables after presolve. The learned SCIP has a "2x to 10x better" average primal-dual gap at large time limits on all datasets but one, where it is "10^5x better". The metric is gap, not time to optimality. — [Nair et al., arXiv 2012.13349](https://arxiv.org/abs/2012.13349)

**RL from scratch**
- Etheve et al. (2020) and Scavuzzo et al. ("Learning to Branch with Tree MDPs", NeurIPS 2022) learn branching without an expert.
  - Tree MDPs give a tree policy-gradient theorem with better credit assignment than temporal MDPs.
  - The tree Markov property is enforced by an optimal objective limit rather than depth-first search.
  - RL converges faster and gives better rules than a regular MDP. — [Scavuzzo et al. 2022, arXiv 2205.11107](https://arxiv.org/pdf/2205.11107v3.pdf); [mlanthology](https://mlanthology.org/neurips/2022/scavuzzo2022neurips-learning)
- Compared with imitation learning, RL wins only where strong branching struggles. Where strong branching works well, Gasse-style imitation is superior. — [Survey](https://arxiv.org/abs/2402.05501v1)
- Benchmarks: set covering, combinatorial auction, maximum independent set and multiple knapsack, with transfer to larger instances. — [WizWand summary](https://www.wizwand.com/paper/68fa59ac1e209131ce9b937e)

**Node selection**
- He et al. (oracle imitation), Yilmaz & Yorke-Smith (learned diving among children), Labassi et al. (a learned pairwise node comparator, GNN) and Khalil et al. (similarity to a predicted solution). The survey's verdict: "the margins of improvement remain small"; heuristic schedules and branching usually matter more than node selection. — [Survey](https://arxiv.org/abs/2402.05501v1)

**Towards one model for many classes**
- Li, Kulkarni, Menache, Wu, Li, "Towards Foundation Models for MILP" (ICLR 2025, Microsoft). MILP-Evolve uses an LLM evolutionary loop to generate diverse MILP classes and trains a single model across classes. Learning to branch (strong-branching imitation, GNN + attention) is one of three tasks. The authors report "significant improvements on unseen problems, including MIPLIB". Code is in microsoft/OptiGuide. — [Microsoft Research page](https://www.microsoft.com/en-us/research/publication/towards-foundation-models-for-mixed-integer-linear-programming/); [mlanthology](https://mlanthology.org/iclr/2025/li2025iclr-foundation)
- Distributional MIPLIB (Huang, Huang, Ferber, Dilkina; arXiv 2406.06954, June/Oct 2024):
  - It is a multi-domain library of MILP distributions graded by hardness, built from prior ML-for-MILP work plus new real-world problems.
  - Branching policies trained on a mix of distributions beat single-distribution training when data is limited, and they generalize to larger instances. — [arXiv 2406.06954](https://arxiv.org/abs/2406.06954v1)
  - A secondary source says it has 13 domains and a pip package; I did not verify this. — [Emergent Mind](https://www.emergentmind.com/topics/distributional-miplib-library)

### Inferences
- The headline speed-ups are modest. On medium instances of the families Gasse trained on, GCNN cuts time by about 1.2-1.7x relative to SCIP default. On hard instances the gain shrinks to about 1.0-1.2x, and RPB still wins on medium MIS.
- "Large" speed-ups (Nair) are measured as primal-dual gap at a time limit on specific distributions, not time to optimality on MIPLIB.
- RL from scratch has not displaced imitation of strong branching. It helps only where strong branching is a poor expert.
- The research frontier in 2025-2026 has moved from per-family models to cross-distribution or "foundation" models. Training data for them is generated by LLMs (MILP-Evolve) or curated (Distributional MIPLIB).

### Gaps
- Gupta et al. (NeurIPS 2020 "Hybrid models for learning to branch"): I found no exact speed-up figure for the CPU-only hybrid in a source I checked.
- Scavuzzo 2022: per-benchmark node or time numbers not retrieved.
- Nair et al.: the per-dataset Neural Branching tree-size and time numbers were not extracted (only the abstract's gap claims).
- Venues of He 2014 (NeurIPS) and Khalil 2016 (AAAI) are from memory; I did not re-check them against the proceedings.
- I found no independent 2025-2026 reproduction that evaluates the foundation-model branching on MIPLIB 2017 with wall-clock time.

## 2. Cut selection, primal heuristics, configuration, warm starts, Predict-and-Search, LLM-based formulation and heuristic evolution

### Takeaway
Learned cut selection improves root bounds under the lab protocols, but realistic multi-cut rounds and final solve time give mixed results. The clearest industrial evidence is Huawei's cut ranker (12.42% online speed-up) and CPLEX's MIQP linearization classifier. Solution-prediction methods (Neural Diving, Predict-and-Search, ConPS) give large primal-gap improvements over SCIP but much smaller ones over Gurobi. From 2024 to 2026 the field shifted toward LLM-driven program evolution: cut selectors, branching rules and diving heuristics evolved as code and tested in SCIP. LLM agents also write the MILP formulations themselves (OptiMUS and successors).

### Cited Findings
**Cut selection**
- Tang, Agrawal & Faenza (2020): the first work to validate learning for cut selection. It is an MDP over Gomory cuts trained with evolution strategies. — [Deza & Khalil cut survey, arXiv 2302.09166](https://arxiv.org/pdf/2302.09166)
- Paulus et al.: imitation learning of a one-step lookahead expert that picks the cut with the largest LP-bound gain.
  - On the bound-improvement metric it beats Tang et al. and SCIP 7.0.2's default rule. The authors show promising solve-time results when integrated in SCIP, but they exclude root-node processing time.
  - Critique: real solvers add many cuts per round, so per-cut LP resolves are unrealistic, and the metric rewards greedy bound gains. — [Survey](https://arxiv.org/abs/2402.05501v1)
- Wang et al. learn the fraction of cuts to add and score ordered subsets with RL, using solve time as reward; this is costly to sample. — [Survey](https://arxiv.org/abs/2402.05501v1)
- Turner et al. learn per-instance weights for SCIP's four cut-scoring functions with RL. Per-instance grid search shows large potential. The learned policy beats random on root gap closed but "does not perform consistently better than SCIP's default settings when it comes to final solving time". — [Survey](https://arxiv.org/abs/2402.05501v1)
- Huang et al., "Learning to Select Cuts for Efficient MIP" (Pattern Recognition 123, 2022; arXiv 2105.13645; SJTU, Huawei, UCL). The cut ranker is trained by multiple-instance learning. On Huawei's production-planning MIP solver it gave a 14.98% solve-time reduction in offline A/B tests and a 12.42% speed-up online, with no loss of accuracy. — [arXiv 2105.13645](https://arxiv.org/pdf/2105.13645)

**Primal heuristics and large neighbourhood search**
- ML-guided LNS that unfixes variables finds better solutions in a fixed time. Even random unfixing often beats running the solver continuously, since solvers spend effort on the dual bound. Liu et al. learn the local-branching neighbourhood size: a supervised initial value plus RL adaptation. This closes the primal gap faster than Fischetti-Lodi's rule and generalizes across sizes and classes. — [Survey](https://arxiv.org/abs/2402.05501v1)
- Han et al., "A GNN-Guided Predict-and-Search Framework for MILP" (ICLR 2023). A GNN predicts marginal probabilities, then a trust region around the prediction is searched with the solver. Primal gap improves by 51.1% vs SCIP and 9.9% vs Gurobi. — [arXiv 2302.05636](https://arxiv.org/pdf/2302.05636); [ICLR 2023 poster](https://iclr.cc/virtual/2023/poster/11194)
- CoCo-MILP (arXiv 2511.09209, Nov 2025) reports a 78.6% gap reduction vs Predict-and-Search (PS) and 63.0% vs Contrastive PS (ConPS) on combinatorial auctions. This is a secondary comparison by a competing method. — [CoCo-MILP, arXiv 2511.09209](https://arxiv.org/html/2511.09209v1)

**Configuration and presolve**
- Bonami, Lodi & Zarpellon, "A Classifier to Decide on the Linearization of MIQPs in CPLEX" (Operations Research 70(6):3303-3320, 2022; conference version 2018). The classifier is "fully integrated and deployed in CPLEX 12.10.0", runs by default, and decides online whether to linearize a convex MIQP. The authors call it, to their knowledge, the first end-to-end integration of ML into a commercial optimization solver. — [RePEc record](https://econpapers.repec.org/RePEc:inm:oropre:v:70:y:2022:i:6:p:3303-3320); [Optimization Online preprint](https://optimization-online.org/wp-content/uploads/2020/03/7662.pdf)
- Li et al. define separator configuration (which separators to activate) as a learning task. — [Survey](https://arxiv.org/abs/2402.05501v1)

**LLM-evolved heuristics (2025-2026)**

Most of these are arXiv preprints. Their numbers are self-reported and there is no common head-to-head benchmark yet.
- DHEvo (arXiv 2507.15615, Jul 2025) co-evolves instances and diving heuristics. It reports a 26.1% better primal-dual integral on LoadBalance vs default SCIP. — [DHEvo](https://arxiv.org/pdf/2507.15615)
- EvoCut (arXiv 2508.11850, Aug 2025): LLM agents plus evolutionary search generate "acceleration cuts", tested against Gurobi with and without internal cuts. — [EvoCut](https://arxiv.org/html/2508.11850v2)
- An IJCAI 2026 paper has an LLM generate stage-aware cut-filtering policies inside SCIP. — [IJCAI 2026 paper 0736](https://www.ijcai.org/proceedings/2026/0736.pdf)
- MILP-Evo (arXiv 2607.18252, Jul 2026) runs closed-loop co-design of a cut selector and a branching rule through PySCIPOpt, scored by end-to-end SCIP runs. — [alphaxiv listing](https://www.alphaxiv.org/abs/2607.18252.md)
- LLM4Branch evolves Python branching programs, tunes their parameters by Bayesian optimization and uses node count as feedback. — [GitHub hzn18/LLM4Branch](https://github.com/hzn18/LLM4Branch)

**LLM-based formulation**
- OptiMUS (Stanford; arXiv 2310.06116 and 2402.10172) is an LLM agent that formulates LP/MILP models, writes and debugs solver code. It solves 67% more problems than basic prompting on its NLP4LP set (40 problems in the first version). The authors report beating prior state of the art by over 20% on existing datasets and by 30% on their own. NL4Opt has 1,101 instances. — [OptiMUS arXiv 2310.06116](https://arxiv.org/html/2310.06116v1); [OptiMUS scalable, ar5iv 2402.10172](https://ar5iv.labs.arxiv.org/html/2402.10172)
- A 2024 autoformulation paper (arXiv 2411.01679) reportedly reaches 92.21% on NL4Opt (ORLM: 85.7%) and 72.2% on ComplexOR (OptiMUS: 66.7%). Secondary summary only. — [alphaxiv overview](https://alphaxiv.org/overview/2411.01679v2)

**Benchmarks and software**
- Ecole (a gym-like library on SCIP) and MIPLearn are the standard tools. ML4CO supplied three datasets (two of 10,000 instances, one of 118). — [Survey](https://arxiv.org/abs/2402.05501v1)

### Inferences
- Learned components often optimize a proxy: root LP bound, node count, or primal gap at a time limit. Several careful studies (Turner; the critique of Paulus) find that the proxy does not translate into faster solve time.
- Primal-side ML (diving, Predict-and-Search, LNS) is the most robust "win". It never affects the exactness proof, so a bad prediction costs only time. The size of the win depends on the baseline: about 51% vs SCIP against about 10% vs Gurobi.
- The 2025-2026 trend is to replace neural policies inside the loop with LLM-written code that is cheap to run, CPU-only and interpretable. This removes the GPU-inference cost that held back GNN policies.

### Gaps
- Contrastive Predict-and-Search (Huang et al., believed ICML 2024): I did not verify the primary paper's numbers or venue.
- Venues as I recall them: Paulus et al. "Learning to Cut by Looking Ahead" ICML 2022; Tang et al. ICML 2020. Neither re-verified; one search snippet suggested ICML 2021 for Paulus, which conflicts.
- No verified numbers for learned warm starts or presolve outside the CPLEX classifier.
- No quantitative results found for Neural LNS (Sonnerat et al. 2021), Khalil 2017 "learning to run heuristics" or Chmiela et al. 2021 heuristic scheduling.

## 3. SAT and CSP: NeuroSAT, NeuroCore, Graph-Q-SAT, learned variable selection in CDCL, G4SATBench, neural guidance in CP, theorem proving

### Takeaway
End-to-end GNN SAT solvers (NeuroSAT line) remain research tools: G4SATBench (TMLR 2024) shows they learn greedy-local-search-like behaviour but not backtracking. Practical gains come from light-touch hybrids:
- one-shot GNN priors that set CDCL phases or activities (NeuroCore, NeuroBack: +5-7% instances solved in Kissat);
- cheap online bandits (LRB, Kissat_MAB), which won SAT Competition main tracks in 2021 and 2022;
- in 2025, an LLM-improved Kissat_MAB that its authors report won the main track.

In CP, RL value-selection (SeaPearl) is not yet competitive with industrial solvers.

### Cited Findings
**One-shot GNN priors for CDCL**
- NeuroCore (Selsam & Bjørner, SAT 2019; arXiv 1903.04671) is a simplified NeuroSAT that predicts unsat-core variables. Its prediction periodically resets the solver's VSIDS activity scores ("periodic refocusing").
  - SATCOMP-2018, 5,000 s: MiniSat solved about 10% more problems (205 vs 187), Glucose 4.1 about 11% more, Z3 about 6% more.
  - Trained on a specific scheduling family, Glucose solved about 20% more in 1 h. — [ar5iv 1903.04671](https://ar5iv.labs.arxiv.org/html/1903.04671); [alphaxiv](https://alphaxiv.org/abs/1903.04671)
- Critique: J. Han (2020) reports that later ablations found NeuroCore's gains "no better than a random baseline" on SATCOMP 2018, and that it needs a GPU and an expensive data pipeline. Han instead predicts glue variables, tested in CaDiCaL. — [arXiv 2007.02559](https://arxiv.org/abs/2007.02559v1)
- NeuroBack (Wang et al., UT Austin; ICLR 2024; arXiv 2110.14053). A GNN trained on backbone-variable phases predicts phases for all variables in one inference before solving, then runs CPU-only. Kissat solves up to 5.2% (SATCOMP-2022) and 7.4% (SATCOMP-2023) more problems, i.e. 43 and 40 extra instances. Training data is DataBack, 120,286 samples. — [arXiv 2110.14053](https://arxiv.org/html/2110.14053v7)

**RL for branching**
- Graph-Q-SAT (Kurin et al., NeurIPS 2020; arXiv 1909.11830) is a Q-learning GNN branching heuristic interfaced with MiniSat. It cuts iterations 2-3x, but the authors note that "more work is needed" to reduce wall-clock time in modern solvers. — [arXiv 1909.11830](https://arxiv.org/abs/1909.11830v2); [NeurIPS 2020 poster](https://nips.cc/virtual/2020/poster/16954)

**Benchmarks and surveys**
- G4SATBench (Li, Guo, Si; arXiv 2309.16941, Sept 2023; TMLR 2024) is the first benchmark for GNN SAT solvers: 7 datasets from 3 families (random, pseudo-industrial, combinatorial). GNNs "can effectively learn a solving strategy akin to greedy local search but struggle to learn backtracking search". — [arXiv 2309.16941](https://arxiv.org/abs/2309.16941); [TMLR listing](https://mlanthology.org/tmlr/2024/li2024tmlr-g4satbench)
- Guo et al., "Machine learning methods in solving the Boolean satisfiability problem", arXiv 2203.04755 (2022), is the ML-for-SAT survey. — [G4SATBench search result referencing it](https://arxiv.org/abs/2309.16941)

**Bandits, the SAT Competition and LLM-improved solvers**
- LRB (learning-rate branching) treats variable selection as a multi-armed bandit (exponential recency weighted average). It is implemented in MiniSat and CryptoMiniSat derivatives. — [Liang et al., Waterloo PDF](https://cs.uwaterloo.ca/~ppoupart/publications/sat/learning-rate-branching-heuristic-SAT.pdf)
- Kissat_MAB uses a bandit at restarts to switch between VSIDS and CHB. It won the SAT Competition 2021 main track. — [arXiv 2404.16387](https://arxiv.org/pdf/2404.16387)
- Kissat-MAB-hywalk won gold in the 2022 main track, and Kissat-MAB-ESA gold in the 2022 anniversary track. — [Chu-Min Li's page (self-reported)](https://home.mis.u-picardie.fr/~cli/EnglishPage.html)
- "AE-Kissat2025-MAB", improved on Kissat_MAB "using LLMs", reportedly won the main track of SAT Competition 2025 "with a wide margin". The page calls it the first SAT Competition win by an LLM-improved solver. — [Chu-Min Li's page (self-reported)](https://home.mis.u-picardie.fr/~cli/EnglishPage.html)
- A CP 2026 (LIPIcs vol. 379) paper frames heuristic selection at restarts as an MAB/RL problem. — [LIPIcs CP 2026.39](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2026.39)

**Constraint programming**
- SeaPearl (Chalumeau, Coulon, Cappart, Rousseau; arXiv 2102.09193; CP 2021) is an open-source Julia CP solver that natively learns value-selection heuristics with deep RL (GNN state). It extends Cappart et al.'s RL+CP proof of concept from DP models to general CP models. The authors state it is not yet competitive with industrial solvers. — [SeaPearl arXiv](https://arxiv.org/pdf/2102.09193)
- A follow-up, "Learning a Generic Value-Selection Heuristic Inside a CP Solver" (arXiv 2301.01913), adds restart-based episodes to fix issues of DFS-based MDPs. — [arXiv 2301.01913](https://arxiv.org/pdf/2301.01913)

### Inferences
- The SAT pattern mirrors MIP: neural guidance pays off only when inference is amortized and kept outside the inner loop (NeuroBack's one-shot CPU inference). Per-decision GNN calls (Graph-Q-SAT) improve iteration counts but not wall-clock time.
- The most successful "learning" in competitive SAT so far is lightweight online learning (bandits), not deep nets. The 2025 LLM-improved win, if confirmed by official results, is the first sign that LLM-driven code evolution reaches competition level.
- NeuroCore's contested ablation is a warning that reported +10% instance gains can fall within run-to-run variance.

### Gaps
- NeuroSAT itself (Selsam et al., ICLR 2019): not re-verified here. From memory it is trained on SAT/UNSAT classification of random SR(n) instances and decodes assignments; no number is cited.
- No official SAT Competition 2025 results page was retrieved, so the AE-Kissat2025-MAB claim rests on the authors' own page.
- Neural theorem proving as search (e.g. AlphaProof, LLM-guided Lean proof search): not researched in this budget.
- No quantitative SeaPearl-vs-OR-Tools/Gecode figures were extracted.

## 4. Competitions and benchmarks: ML4CO 2021, MIPLIB, Distributional MIPLIB, ML entries in SAT competitions

### Takeaway
ML4CO (NeurIPS 2021) had three tasks (primal, dual, configuration) on SCIP via Ecole. Winners came largely from industry labs (Huawei, Megvii) and universities in China. Benchmarks have since moved toward multi-domain distributions (Distributional MIPLIB 2024, MILP-Evolve 2025). In SAT competitions, ML has meant bandits, plus an LLM-assisted solver in 2025.

### Cited Findings
**ML4CO (NeurIPS 2021 Competitions track)**
- Three tasks, all on SCIP: the best feasible solution (primal), the tightest optimality certificate (dual), and a good solver configuration. Results paper: Gasse et al., PMLR 176:220-231 (2022); repository ds4dm/ml4co-competition. — [arXiv 2203.02433](https://arxiv.org/pdf/2203.02433); [PMLR](https://proceedings.mlr.press/v176/gasse22a.html); [GitHub](https://github.com/ds4dm/ml4co-competition)
- Winners:
  - Primal: team led by Linxin Yang (CUHK, Huawei and others).
  - Dual: Zixuan Cao and Yang Xu (Peking University) with Zhewei Huang and Shuchang Zhou (Megvii); Huawei EI-OROAS placed a close second.
  - Configuration: Binbin Chen, Minggui He, Zhiyu Zhang, Zhiwu An, Kun Mao and others.

  — [arXiv 2203.02433](https://arxiv.org/pdf/2203.02433)
- Datasets: two homogeneous sets of 10,000 instances and one of 118. — [Survey](https://arxiv.org/abs/2402.05501v1)

**MIP benchmarks**
- More than 68% of MIPLIB instances are purely binary. In the rest, more than 90% of integer variables are binary. — [Survey](https://arxiv.org/abs/2402.05501v1)
- Distributional MIPLIB (2024) and MILP-Evolve (ICLR 2025) are the multi-domain successors; details in section 1. — [arXiv 2406.06954](https://arxiv.org/abs/2406.06954v1); [MSR](https://www.microsoft.com/en-us/research/publication/towards-foundation-models-for-mixed-integer-linear-programming/)

**SAT benchmarks and competitions**
- SAT Competition main-track wins by MAB-based Kissat variants (2021, 2022) and the LLM-improved 2025 entry: see section 3. — [arXiv 2404.16387](https://arxiv.org/pdf/2404.16387); [C.-M. Li page](https://home.mis.u-picardie.fr/~cli/EnglishPage.html)
- G4SATBench is the GNN-SAT benchmark. — [arXiv 2309.16941](https://arxiv.org/abs/2309.16941)

### Inferences
- ML4CO's homogeneous, large training sets match the setting where imitation learning works. MIPLIB's heterogeneity is the setting where it fails. This is why the field built distributional libraries rather than testing on MIPLIB.

### Gaps
- ML4CO's quantitative score margins (e.g. the dual-integral reduction vs SCIP default) were not extracted.
- I did not check whether a successor ML4CO-style competition ran in 2023-2026.

## 5. Adoption by commercial and open solvers, real deployments, and critiques

### Takeaway
As of 2025-2026, no major commercial MIP solver (Gurobi, CPLEX, Xpress) is documented as shipping learned branching or cut selection in its default path. The documented deployments are narrow decisions:
- CPLEX 12.10's MIQP linearization classifier;
- Huawei's in-house cut ranker;
- Google/DeepMind's internal Neural Diving and Branching on production datasets.

Gurobi's ML work is about optimizing over trained models (Gurobi Machine Learning; faster solves of models with embedded ReLU networks), not about ML inside the search. The main critiques are distribution dependence, small or proxy-only gains, inference overhead, and reproducibility.

### Cited Findings
**Commercial solvers**
- An Aug 2025 survey (arXiv 2508.06906) states that as of 2025 none of the widely used commercial solvers, Gurobi included, ship end-to-end ML-based branching or cut selection in official releases. Learning methods remain "prototypes" despite gains on research benchmarks. — [arXiv 2508.06906](https://arxiv.org/pdf/2508.06906)
- CPLEX 12.10.0 deploys an ML classifier by default for MIQP linearization, the first end-to-end ML integration in a commercial solver according to the authors. — [Optimization Online](https://optimization-online.org/wp-content/uploads/2020/03/7662.pdf); [RePEc](https://econpapers.repec.org/RePEc:inm:oropre:v:70:y:2022:i:6:p:3303-3320)
- Gurobi:
  - The 12.0 notes cite over 10x improvement on certain models containing embedded ReLU networks.
  - The 13.0 notes list conventional heuristic improvements, e.g. NoRel with variable hints.
  - Gurobi Machine Learning (2022, with Gurobi 10) embeds trained models as constraints.

  None of these is learned search inside the solver. — [Gurobi release-note summary via search](https://www-old.gurobi.com/?p=12358); [Gurobi INFORMS 2023 workshop slides](https://cdn.gurobi.com/wp-content/uploads/Gurobi_INFORMSAnalytics2023_Workshop_Gurobi10.pdf)

**Deployments**
- Huawei's production MIP solver: the learned cut selection was A/B tested, with 12.42% online speed-up. — [arXiv 2105.13645](https://arxiv.org/pdf/2105.13645)
- DeepMind/Google: Neural Diving and Branching were evaluated on two Google production datasets. — [arXiv 2012.13349](https://arxiv.org/abs/2012.13349)

**Critiques**
- Distribution dependence: Gasse-style models struggle on heterogeneous distributions, and reliability branching still beats tree-context models. — [Survey](https://arxiv.org/abs/2402.05501v1)
- Small margins in node selection, and cut learning that improves root bounds but not solve time (Turner). — [Survey](https://arxiv.org/abs/2402.05501v1)
- Unrealistic single-cut protocols and omitted root times (Paulus). — [Survey](https://arxiv.org/abs/2402.05501v1)
- Expressivity vs speed of models. — [Survey](https://arxiv.org/abs/2402.05501v1)
- SAT: NeuroCore's gains were not distinguishable from a random baseline in later ablations. — [arXiv 2007.02559](https://arxiv.org/abs/2007.02559v1)
- SAT: Graph-Q-SAT has no wall-clock gains. — [arXiv 1909.11830](https://arxiv.org/abs/1909.11830v2)
- SAT: GNNs fail to learn backtracking. — [G4SATBench](https://arxiv.org/abs/2309.16941)
- Gains depend on the baseline: Predict-and-Search improves the primal gap 51.1% vs SCIP but only 9.9% vs Gurobi, consistent with Gurobi's stronger built-in primal heuristics. — [arXiv 2302.05636](https://arxiv.org/pdf/2302.05636); [Liner review](https://liner.com/review/a-gnnguided-predictandsearch-framework-for-mixedinteger-linear-programming)

### Inferences
- Deployed ML targets one-time, low-overhead, low-risk decisions (linearize or not; which cuts to keep) where the training distribution is the vendor's or customer's own recurring workload. This matches the "same class solved routinely" premise that Gasse et al. used to justify per-class models.
- By late 2026 the momentum (LLM-evolved heuristics, the LLM-improved SAT winner, LLM formulation agents) points to LLM-written code and modelling assistance being adopted earlier than GNN policies inside commercial B&B.
- For any domain-specific exact search (including mechanism-search problems such as the user's), the transferable lessons are:
  1. amortize neural inference (one-shot priors over per-node calls);
  2. use ML for primal guidance and ordering, which never compromises exactness;
  3. evaluate on wall-clock time to a proven optimum against the strongest default, not on proxies.

### Gaps
- No primary 2026 documentation found for Hexaly, OR-Tools CP-SAT, FICO Xpress or SCIP 9/10 on built-in learned components. I could not confirm or rule out ML features there.
- Gurobi 13.0's release date and full notes were not retrieved directly.
- I found no independent large-scale reproduction study (e.g. 2024-2026) quantifying ML branching wall-clock gains on MIPLIB 2017 against a tuned default.
