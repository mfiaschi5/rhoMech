# Ecosystem of deep learning for combinatorial optimization: libraries, benchmarks, surveys, critiques (as of Nov 2026)

Research date: 2026-11-08. Web research only. Sources were found with about 17 search and fetch calls. Where a classic paper is cited by its arXiv identifier from prior knowledge rather than a page fetched in this session, it is marked "(ID from prior knowledge)". The report writer should treat those as reliable bibliographic pointers but not as freshly verified content.

## Libraries and frameworks: what exists, how mature, maintained or not

### Takeaway
The open-source stack has three layers: (1) learning frameworks for neural solvers (RL4CO in PyTorch; Jumanji in JAX; ML4CO-Kit from SJTU, which also underlies several benchmarks), (2) gym-style interfaces into exact solvers (Ecole for SCIP, now only maintained for critical issues), and (3) strong classical solvers used as baselines or as hosts for learned components (PyVRP/HGS, LKH-3, OR-Tools, SCIP/PySCIPOpt, Gurobi with its ML-embedding package). RL4CO is the most visible neural-CO library (about 850 GitHub stars in early 2026). Ecole is the main case of an abandoned key tool.

### Cited Findings
**RL4CO (ai4co/rl4co)**
- A PyTorch library for reinforcement learning on combinatorial optimization. Paper: arXiv 2306.17100 (Berto et al.) — [arXiv](https://arxiv.org/abs/2306.17100); [GitHub](https://github.com/ai4co/rl4co)
- Star counts disagree by snapshot: 847 stars as of 2026-04-09, growing about 1 to 2 stars per day ([oosmetrics](https://oosmetrics.com/repo/ai4co/rl4co)); 859 stars, last push 2026-01-22 ([gittrend](https://gittrend.io/repo/ai4co/rl4co)); 631 with no date ([ecosyste.ms](https://repos.ecosyste.ms/hosts/GitHub/repositories/ai4co%2Frl4co)); 512 in early 2025 ([repositorystats](https://repositorystats.com/ai4co/rl4co)).

**ML4CO-Kit (Thinklab-SJTU)**
- A Python toolkit (`pip install ml4co-kit`). It provides base classes for solvers, mainstream traditional solvers, decoding and post-processing algorithms, evaluators, data generators, visualization, and read/write functions. Its learning base classes are built on PyTorch Lightning. — [docs](https://ml4co-kit.readthedocs.io/en/stable/); [PyPI](https://pypi.org/project/ml4co-kit/)
- It is the shared base of three benchmarks: ML4CO-Bench-101 (classifies neural CO solvers by solving paradigm, model design and learning strategy), ML4TSPBench (TSP), and PredictiveCO-Benchmark (decision-focused learning). Python 3.8 support ends at version 0.5.3; ML4CO-Bench-101 pins ml4co-kit==0.3.3. — [docs](https://ml4co-kit.readthedocs.io); [ML4TSPBench GitHub](https://github.com/Thinklab-SJTU/ML4TSPBench); [ML4CO-Bench-101 (ecosyste.ms)](https://awesome.ecosyste.ms/projects/github.com%2FThinklab-SJTU%2FML4CO-Bench-101)

**Ecole (ds4dm/ecole)**
- A gym-like library for machine learning inside combinatorial optimization solvers (SCIP). Paper: arXiv 2011.06069. — [arXiv](https://arxiv.org/abs/2011.06069?context=cs); [GitHub](https://github.com/ds4dm/ecole)
- Maintenance: upstream notice (as quoted in a fork's README) says Ecole is "looking for a new home", "not being actively developed", and "only critical issues will be investigated". An ecosyste.ms record shows the repo unarchived, with the last push in February 2023. Community forks exist on PyPI (`ecole-fork` 0.9.x). — [fork README](https://github.com/FreedomLLL/ecole); [ecosyste.ms record](https://awesome.ecosyste.ms/api/v1/projects/github.com%2Fds4dm%2Fecole); [PyPI ecole-fork](https://pypi.org/project/ecole-fork/0.9.4/)

**PyVRP (classical baseline and host)**
- Wouda, Lan and Kool, "PyVRP: A High-Performance VRP Solver Package", INFORMS Journal on Computing 36(4):943-955, August 2024, DOI 10.1287/ijoc.2023.0055. It implements hybrid genetic search (HGS), building on Vidal's HGS-CVRP with time windows added. The C++ core is wrapped in Python. MIT license, `pip install pyvrp`. — [VU research portal](https://research.vu.nl/en/publications/pyvrp-a-high-performance-vrp-solver-package/); [arXiv 2403.13795](https://arxiv.org/pdf/2403.13795v2.pdf); [RePEc](https://econpapers.repec.org/RePEc:inm:orijoc:v:36:y:2024:i:4:p:943-955)
- The underlying algorithm ranked 1st in the 2021 DIMACS VRPTW challenge. The package ranked 1st on the static variant of EURO meets NeurIPS 2022. — [arXiv 2403.13795](https://arxiv.org/pdf/2403.13795v2.pdf)

**Classical solvers that serve as baselines in recent evaluations** (as used in FrontierCO, 2025)
- KaMIS (MIS), LKH-3 (TSP), HGS (CVRP), GB21-MH (capacitated p-median), SCIP-Jack (Steiner tree), Gurobi (MDS, CFLP), CPLEX (FJSP). — [FrontierCO arXiv 2505.16952](https://arxiv.org/html/2505.16952v2)
- The 2026 neural-routing survey argues that HGS and LKH-3 are no longer state of the art in the heuristics literature, where AILS-II is more competitive: on its CVRP set AILS-II gets 0.68% vs 0.85% for HGS. — [Survey on Neural Routing Solvers, arXiv 2602.21761](https://arxiv.org/html/2602.21761v1)

**Other libraries named in the brief** (pointers only; not verified this session)
- Jumanji (InstaDeep): JAX-based RL environments including combinatorial ones (TSP, CVRP, knapsack, job shop, bin packing, etc.); paper arXiv 2306.09884, ICLR 2024 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/2306.09884)
- OR-Gym: RL environments for operations research, Hubbs et al. 2020, arXiv 2008.06319 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/2008.06319)
- DIMES (arXiv 2210.04123) and DIFUSCO (arXiv 2302.08224) release research code; DIFUSCO is a standard baseline in later diffusion-based CO work, and FrontierCO evaluates it (IDs from prior knowledge) — [DIMES](https://arxiv.org/abs/2210.04123); [DIFUSCO](https://arxiv.org/abs/2302.08224); [FrontierCO](https://arxiv.org/html/2505.16952v2)
- OR-Tools (Google), PySCIPOpt (SCIP's Python interface), and Gurobi Machine Learning (`gurobi-machinelearning`, embeds trained regression and neural-network models as MIP constraints) — [OR-Tools](https://github.com/google/or-tools); [PySCIPOpt](https://github.com/scipopt/PySCIPOpt); [gurobi-machinelearning](https://github.com/Gurobi/gurobi-machinelearning)
- PyEPO (Tang and Khalil, arXiv 2206.14234): PyTorch library for predict-then-optimize and decision-focused learning. The Mandi et al. survey describes it as "a common code base and an implementation of a selected number of DFL techniques" with a common benchmarking interface — [Mandi et al. arXiv 2307.13565](https://arxiv.org/html/2307.13565v4)
- Code for FrontierCO and CO-Bench (classical solvers, best-known-solution computation, LLM agents) — [FrontierCO GitHub](https://github.com/sunnweiwei/FrontierCO); [CO-Bench GitHub](https://github.com/sunnweiwei/CO-Bench)

### Inferences
- RL4CO is the most visible neural-CO library, but at about 850 stars it is small next to general ML tooling. It is a research codebase, not industrial software.
- Ecole is unmaintained, which leaves a gap for learned branching and solver-control research on SCIP. Researchers now likely write custom PySCIPOpt callbacks or use forks; this is not verified.
- Recent evaluations treat the classical solvers (PyVRP/HGS, LKH-3, KaMIS, Gurobi) as reference points that neural methods should be measured against, not as tools to be replaced.

### Gaps
- No verified 2026 star counts or maintenance status this session for Jumanji, OR-Gym, ML4CO-Kit, DIFUSCO/DIMES, PyVRP or PySCIPOpt. OR-Gym appears dormant (prior knowledge, unverified).
- "Routing gyms" (e.g., RouteFinder, the EURO-NeurIPS VRP environment) were not verified.
- PyTorch Geometric and DGL usage in CO was not measured. Prior knowledge: DGL development slowed after 2024, and PyG is the more common GNN backbone in CO code (unverified).
- Whether Gurobi has added ML features beyond `gurobi-machinelearning` by late 2026 was not checked.

## Benchmarks and datasets: what is standard

### Takeaway
Classical libraries (TSPLIB, CVRPLIB X set, MIPLIB 2017, SATLIB and SAT Competition sets, Taillard JSSP, OR-Library, DIMACS) remain the ground truth. ML-specific suites have grown on top of them: the ML4CO 2021 competition data, Distributional MIPLIB (2024), G4SATBench, ML4CO-Bench-101, and in 2025 the larger FrontierCO. A separate family of benchmarks for LLMs on optimization appeared in 2023-2025: NL4Opt, OptiBench and OptMATH test modeling a problem from text; CO-Bench, ALE-Bench and FrontierCO test algorithm design. Known issues in this family: label errors, and benchmark sizes that differ between papers.

### Cited Findings
**Classical instance libraries**
- TSPLIB (Reinelt 1991): standard symmetric/asymmetric TSP instances with best-known solutions — [TSPLIB95](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) (URL from prior knowledge)
- MIPLIB 2017: the reference MIP library (collection set and a 240-instance benchmark set, from prior knowledge) — [MIPLIB 2017](https://miplib.zib.de/)
- SATLIB: classic SAT benchmark collection — [SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/) (URL from prior knowledge)
- FrontierCO draws its instances from TSPLib, DIMACS challenges, BHOSLib, PACE 2025, CVRP and CFLP testbeds, CPMP and FJSP repositories, and Steiner tree benchmarks. Each problem has an "easy" set (solved by state-of-the-art classical methods) and a "hard" set (open or computationally heavy). Sizes reach TSP with 10M cities and MIS with about 8M nodes. — [FrontierCO](https://arxiv.org/html/2505.16952v2)
- The 2026 neural-routing survey's evaluation uses TSPLIB, National, VLSI, DIMACS-challenge and CVRPLIB instances up to 100K nodes: 228 TSP instances and 110 CVRP instances. — [arXiv 2602.21761](https://arxiv.org/html/2602.21761v1)

**ML-specific MIP and SAT sets**
- Distributional MIPLIB (Huang, Huang, Ferber, Dilkina; USC and Cornell; arXiv 2406.06954, June 2024): MILP instance distributions from 13 domains (10 synthetic, 3 real-world not used before). Each distribution has one of 5 hardness levels, set from Gurobi 10.0.3 statistics under a 1-hour limit. It ships a pip package `distributional_miplib` and a CLI. The paper finds that branching policies trained on a mix of distributions beat those trained on one distribution when data is limited, and generalize to larger instances. — [arXiv](https://www.arxiv.org/pdf/2406.06954); [site](https://sites.google.com/usc.edu/distributional-miplib)
- ML4CO NeurIPS 2021 competition: three tasks on SCIP via Ecole (primal task, dual task, configuration task), with three instance datasets (item placement, load balancing, anonymous MIPLIB-like). Report: Gasse et al., arXiv 2203.02433 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/2203.02433)
- G4SATBench: benchmark for GNNs on SAT (Li, Guo, Si; arXiv 2309.16941, from prior knowledge; not verified this session) — [arXiv](https://arxiv.org/abs/2309.16941)

**Benchmarks for LLMs as algorithm designers**
- CO-Bench (Sun, Feng, Li, Yang; CMU; arXiv 2504.04310; published at AAAI 2026): 36 problems in 8 categories, mostly from OR-Library (packing, cutting, facility location, scheduling, routing, assignment, tree, graph, set). It evaluated 15 LLMs and 9 agent frameworks against classical algorithms and best-known solutions, with a 10 s single-CPU limit per instance. Reasoning models beat base models, and FunSearch-style frameworks helped further. Remaining limits: planning, feasibility checking, and efficiency of the generated code. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40596); [arXiv](https://arxiv.org/html/2504.04310v2)
- ALE-Bench (Sakana AI with AtCoder; arXiv 2506.09050, June 2025): 40 past AtCoder Heuristic Contest problems, plus a 10-problem lite set. Scores are converted to an Elo-like performance rating against human contestants, and solutions are submitted iteratively in a sandbox. Sakana's ALE-Agent (reported to run on Gemini 2.5 Pro) placed 21st of about 1,000 humans in AHC047 (May 2025); this claim comes mainly from Sakana and the press. — [arXiv](https://arxiv.org/pdf/2506.09050); [Sakana](https://sakana.ai/ale-bench/); [Epoch AI](https://epoch.ai/benchmarks/ale-bench); [The Decoder](https://the-decoder.com/sakana-ais-ale-ai-agent-cracks-the-top-21-among-1000-code-experts/)
- FrontierCO (Feng, Sun, Li, Talwalkar, Yang; CMU; arXiv 2505.16952): 8 problems (MIS, MDS, TSP, CVRP, CFLP, CPMP, FJSP, STP). Dataset on Hugging Face `CO-Bench/FrontierCO`. — [arXiv](https://arxiv.org/html/2505.16952v2); [HF dataset](https://huggingface.co/datasets/CO-Bench/FrontierCO)

**Benchmarks for modeling optimization problems from natural language**
- NL4Opt: NeurIPS 2022 competition dataset of linear programming word problems. Competition report arXiv 2303.08233 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/2303.08233)
- OptiBench (ICLR 2025; arXiv 2407.09887): 605 manually verified problems, linear and nonlinear, integer and mixed-integer, some with tabular data. Evaluation is end-to-end: the model calls a solver, and a problem counts as solved only if the variables and objective match the numerical ground truth. — [arXiv](https://arxiv.org/html/2407.09887v4); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/3deb687c44d3687ace0729e5db3b4efd-Paper-Conference.pdf)
- OptMATH (arXiv 2502.11102): built because NL4Opt and MAMO contain only linear programs. Its "hard" benchmark comes from instances its synthesis pipeline rejected. The authors note that ambiguous statements make integer vs continuous variables hard to decide. — [arXiv](https://arxiv.org/pdf/2502.11102)
- Data quality: later work (Solver-Informed RL, arXiv 2505.11792) reports errors in test sets and manually corrected NL4Opt and IndustryOR. Reported sizes differ between papers (NL4Opt as 213, 245 or 1,101 instances; OptMATH-Bench as 166 or 193). — [arXiv 2505.11792](https://arxiv.org/pdf/2505.11792); [arXiv 2605.02545](https://arxiv.org/pdf/2605.02545)

### Inferences
- Two benchmark families now coexist. One is synthetic: uniform random TSP/CVRP of 20 to 100 nodes, the Kool et al. (2019) setup. The other is realistic and large-scale: TSPLIB/CVRPLIB/FrontierCO. Results on the first do not transfer to the second (see the critiques section).
- For LLM optimization-modeling benchmarks, the first thing to check is which corrected version of NL4Opt/IndustryOR a paper used. Otherwise numbers are not comparable.

### Gaps
- Not verified this session: CVRPLIB X instance details (Uchoa et al. 2017, 100 instances of 100 to 1000 customers, prior knowledge), Taillard JSSP (1993), SAT Competition tracks, Max-Cut/MIS graph sets (Gset, SATLIB-derived MIS, ER/BA graphs used in DIMES/DIFUSCO).
- No authoritative list of "standard" ML4CO datasets exists. Papers with Code leaderboards for CO are fragmented. Papers with Code was shut down by Meta in July 2025 (prior knowledge, unverified), which affects any recommendation to use it.

## Surveys, books, tutorials and courses

### Takeaway
The canonical reading list is: Bengio/Lodi/Prouvost (EJOR 2021), Cappart et al. (JMLR 2023), Mazyavkina et al. (C&OR 2021), Kotary et al. (IJCAI 2021) for end-to-end constrained learning, and Mandi et al. (JAIR 2024) for decision-focused learning. Since 2025 there are new surveys on LLMs for CO/OR: Da Ros et al. in ACM Computing Surveys (April 2026), and a May 2026 survey of LLMs for OR. There is also a 2026 survey of neural routing solvers that covers 344 methods and includes its own critical benchmark.

### Cited Findings
- Bengio, Lodi, Prouvost, "Machine learning for combinatorial optimization: a methodological tour d'horizon", EJOR 2021; arXiv 1811.06128 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/1811.06128)
- Cappart, Chételat, Khalil, Lodi, Morris, Veličković, "Combinatorial optimization and reasoning with graph neural networks", JMLR 24 (2023); arXiv 2102.09544 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/2102.09544)
- Mazyavkina, Sviridov, Ivanov, Burnaev, "Reinforcement learning for combinatorial optimization: a survey", Computers & OR 2021; arXiv 2003.03600 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/2003.03600)
- Kotary, Fioretto, Van Hentenryck, Wilder, "End-to-End Constrained Optimization Learning: A Survey", IJCAI 2021; arXiv 2103.16378 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/2103.16378)
- Mandi, Kotary et al., "Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities", JAIR vol. 80, 2024, doi:10.1613/JAIR.1.15320. Splits gradient-based from gradient-free DFL, and evaluates 11 methods on 7 problems. Code at github.com/PredOpt/predopt-benchmarks. — [arXiv 2307.13565](https://arxiv.org/html/2307.13565v4); [mlanthology](https://mlanthology.org/jair/2024/mandi2024jair-decisionfocused)
- Da Ros, Soprano, Di Gaspero, Roitero, "Large Language Models for Combinatorial Optimization: A Systematic Review": arXiv July 2025, published in ACM Computing Surveys 58(11), Article 272, April 2026. It screened more than 2,000 publications and kept 103. Taxonomy: task types, architectures, benchmarks, applications. Key limitations it reports: data scarcity, poor generalization across problem classes, weak interpretability. — [arXiv 2507.03637](https://arxiv.org/pdf/2507.03637); [Uniud repository](https://air.uniud.it/handle/11390/1325705)
- "Large Language Models for Operations Research: A Comprehensive Survey", posted 20 May 2026 (math.OC). Highlights scalability, data heterogeneity and explainability. — [Emergent Mind entry 2605.20849](https://www.emergentmind.com/papers/2605.20849) (secondary listing; primary arXiv page not fetched)
- Ba, Lin, Zhou, ..., Zhang, "Survey on Neural Routing Solvers", arXiv 2602.21761 (25 Feb 2026). 344 methods surveyed up to Nov 2025. Taxonomy: construction (single-stage appending or insertion; two-stage), improvement with a single solution (small or large neighbourhood), and improvement with a population (discrete such as DeepACO/GFACS; continuous latent such as COMPASS). — [arXiv](https://arxiv.org/html/2602.21761v1)
- A 2024 VRP-focused NCO survey (arXiv 2406.00415) lists poor generalization across distributions and scales as a core limitation. — [arXiv](https://web3.arxiv.org/pdf/2406.00415v3)
- Foundational methods papers often used as tutorial anchors: Kool, van Hoof, Welling, "Attention, Learn to Solve Routing Problems!", ICLR 2019, arXiv 1803.08475; Gasse et al., "Exact combinatorial optimization with graph convolutional neural networks", NeurIPS 2019, arXiv 1906.01629 (IDs from prior knowledge) — [Kool](https://arxiv.org/abs/1803.08475); [Gasse](https://arxiv.org/abs/1906.01629)
- Workshop: Learning Meets Combinatorial Algorithms (LMCA), NeurIPS 2020 — [site](https://sites.google.com/view/lmca2020) (URL from prior knowledge, not fetched)

### Inferences
- From the 2026 surveys on, attention has shifted from GNNs and RL toward LLM-driven algorithm discovery (FunSearch, EoH, ReEvo) and toward stricter evaluation.
- For a newcomer, a good order is: Bengio et al. (framing), Cappart et al. (GNNs), the 2026 routing survey (current state plus honest benchmark), Mandi et al. (DFL), Da Ros et al. (LLMs).

### Gaps
- "Berto et al." survey: apart from the RL4CO paper, I could not confirm a separate Berto et al. survey this session.
- Tutorials and courses (NeurIPS/IJCAI/AAAI tutorials on ML for CO, the CPAIOR master classes, the Learning to Optimize community) were not verified. The 2025-2026 workshop series at NeurIPS/ICLR on ML and CO or LLMs for optimization was not checked.
- No recent textbook dedicated to ML for CO was found.

## Critiques and open problems: generalization, fair baselines, evaluation pitfalls

### Takeaway
There is a consistent body of critique. Neural CO results on small uniform synthetic instances do not survive realistic evaluation. Several reasons recur: generalization to larger or out-of-distribution instances breaks down; baselines are weak or badly tuned; the GNN often contributes little inside hybrid pipelines; and time and hardware accounting is inconsistent. The 2025-2026 benchmarks (FrontierCO, the routing survey) put numbers on this. LEHD goes from a 0.72% gap on the standard benchmark to about 10% on easy and 77% on hard FrontierCO TSP. Many neural routing solvers do worse than nearest neighbour on medium and large instances. The best neural routing solver reaches 1.71% average gap on TSP, against 0.03% for LKH-3.

### Cited Findings
**Graph-level MIS critiques**
- Angelini and Ricci-Tersenghi (arXiv 2206.13211; Nature Machine Intelligence comment, 2023) respond to the physics-inspired GNN of Schuetz et al. (Nat Mach Intell 4, 367, 2022). A near-linear-time greedy algorithm finds better maximum independent sets than the GNN, and is about 10^4 times faster on graphs with a million variables. Belief propagation with reinforcement also beats the GNN. They propose greedy as the minimum baseline to beat. — [arXiv](https://arxiv.org/pdf/2206.13211)
- Schuetz et al. replied (Nat Mach Intell 5, 32, 2023): the comment looks at one non-representative problem, and random regular graphs are not a universal benchmark. They report that GraphSAGE reaches approximation ratios comparable to greedy. — [arXiv 2302.03602](https://export.arxiv.org/abs/2302.03602v1)
- Böther, Kißig, Taraz, Cohen, Seidel, Friedrich, "What's Wrong with Deep Learning in Tree Search for Combinatorial Optimization", ICLR 2022 (arXiv 2201.10494). In a GNN-guided tree search for MIS, the GCN can be replaced by random values without loss; the performance comes from classical kernelization. Classical solvers are often faster at similar quality. The authors also released an open benchmark suite. For one RL solver, by contrast, the GNN does drive solution quality. — [arXiv](https://arxiv.org/abs/2201.10494v1); [ICLR](https://iclr.cc/virtual/2022/poster/6426)

**Routing generalization**
- Joshi, Cappart, Rousseau, Laurent, "Learning TSP requires rethinking generalization", CP 2021, arXiv 2006.07054 (ID from prior knowledge): end-to-end neural TSP solvers trained on small instances degrade sharply on larger ones — [arXiv](https://arxiv.org/abs/2006.07054)
- FrontierCO (2025) evaluation protocol: 1 h per instance; classical solvers on one CPU core of a dual AMD EPYC 7313; neural solvers on one RTX A6000. Metric is primal gap; infeasible runs count as gap 1 and 3600 s. — [arXiv 2505.16952](https://arxiv.org/html/2505.16952v2)
- FrontierCO results: neural solvers ran out of memory on 4 of 8 problems. LEHD gave no solution on the largest TSP in time. Neural components do improve weak heuristics (DIFUSCO 4.19% vs 2-opt 20.09% on easy TSP; GCNN 3.22% vs SCIP 6.50% on easy CFLP) but stay behind LKH-3 (0.03%) and Gurobi (0.00%). A GNN predicts Steiner nodes well on Euclidean graphs and fails on non-Euclidean ones. Average gaps of LLM agents: FunSearch 20.35%, Self-Refine 15.11%, ReEvo 13.25%. Some wins: Self-Refine beat KaMIS on easy MIS, FunSearch beat HGS on hard CVRP. LLM agents were weak on TSP, and their algorithms often reduce to simulated annealing or LNS. — [arXiv 2505.16952](https://arxiv.org/html/2505.16952v2)
- Survey on Neural Routing Solvers (Feb 2026) criticizes the conventional pipeline for: narrow, uniform test distributions; reporting multi-model in-distribution results as if they were generalization; inconsistent inference settings (beam search, augmentation, active search); weak or badly tuned heuristic baselines. Its protocol: one model per solver, zero-shot, greedy inference only, with gap, time and reliability (fraction of instances solved). — [arXiv 2602.21761](https://arxiv.org/html/2602.21761v1)
- Numbers under that protocol, TSP (228 instances): LKH-3 0.03%; DRHG 1.71% (best neural); L2R 8.06%; SIL 9.75%; BQ 14.02% (164/228 solved); random insertion 14.12%; LEHD 16.19% (186/228); nearest neighbour 25.88%; Fast T2T solved 45/228. CVRP (110): AILS-II 0.68%; HGS 0.85%; ReLD 4.58%; ICAM 5.32% (104/110); L2R 8.48% (110/110); nearest neighbour 20.39%. Under the conventional pipeline, ICAM with augmentation reaches about 0.15% on TSP100. GFACS goes from 2.38% to 45.42% on CVRP without its final local search. Only L2R, SIL (PRC) and DRHG solve every instance. — [arXiv 2602.21761](https://arxiv.org/html/2602.21761v1)
- The same survey's position: poor generalization most likely comes from narrow training distributions, not a fundamental limit of deep learning. — [arXiv 2602.21761](https://arxiv.org/html/2602.21761v1)

**LLM side**
- The Da Ros et al. systematic review names data scarcity, weak generalization across problem classes, and limited interpretability as the main open problems for LLMs on CO. — [arXiv 2507.03637](https://arxiv.org/pdf/2507.03637)
- CO-Bench: agents still struggle with planning, feasibility checking and writing efficient code. — [arXiv 2504.04310](https://arxiv.org/html/2504.04310v2)
- ALE-Bench: top LLMs do well on single tasks but fall behind human experts on generalization across tasks and robustness over long horizons. — [ALE-Bench summary (newx.sg)](https://newx.sg/paper/detail/87562ca3-f823-11f0-b84c-00163e10baa7); [arXiv](https://arxiv.org/pdf/2506.09050)
- Optimization-modeling benchmarks have label errors (NL4Opt, IndustryOR corrected by later authors) and inconsistent sizes between papers. — [arXiv 2505.11792](https://arxiv.org/pdf/2505.11792); [arXiv 2605.02545](https://arxiv.org/pdf/2605.02545)

### Inferences
- A defensible evaluation in late 2026 needs, at minimum:
  - realistic or structured instances, not only uniform random ones (TSPLIB/CVRPLIB/FrontierCO hard sets);
  - one model across sizes, with zero-shot generalization;
  - state-of-the-art classical baselines with sensible settings (LKH-3, HGS or AILS-II, KaMIS, Gurobi), plus simple greedy baselines;
  - equal time budgets, reported with the hardware (CPU core vs GPU);
  - reliability or failure rates;
  - ablations that test whether the learned component matters (random-replacement tests as in Böther et al.; removing local search as with GFACS).
- In the strongest current results the learned part improves or guides a classical search (neural-guided LNS/LKH, learned branching), rather than producing solutions end to end. This is consistent with both FrontierCO and the routing survey.

### Gaps
- Not fetched this session: Boettcher's Max-Cut critique of the same physics-inspired GNN (Nature Machine Intelligence 2023, prior knowledge), and later rebuttals.
- No unified, community-maintained leaderboard with hardware-normalized timing was found.
- A systematic study of reproducibility (re-running published neural CO code) beyond FrontierCO and the routing survey was not found.

## Related line: differentiable optimization layers and decision-focused learning

### Takeaway
A separate but adjacent line makes the solver itself part of the network, by differentiating through a convex or combinatorial optimization layer. Key papers: OptNet (QP layers), CvxpyLayers (disciplined convex programs), blackbox differentiation of combinatorial solvers, and perturbed optimizers. Decision-focused (predict-then-optimize) learning builds on these layers. It is surveyed and benchmarked by Mandi et al. (JAIR 2024), with software in PyEPO, PredOpt benchmarks and ML4CO-Kit's PredictiveCO-Benchmark.

### Cited Findings
- OptNet: Amos and Kolter, ICML 2017, differentiable quadratic-program layers, arXiv 1703.00443 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/1703.00443)
- CvxpyLayers: Agrawal et al., "Differentiable Convex Optimization Layers", NeurIPS 2019, arXiv 1910.12430 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/1910.12430)
- Blackbox differentiation: Vlastelica (Pogančić) et al., "Differentiation of Blackbox Combinatorial Solvers", ICLR 2020, arXiv 1912.02175 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/1912.02175)
- Perturbed optimizers: Berthet et al., "Learning with Differentiable Perturbed Optimizers", NeurIPS 2020, arXiv 2002.08676 (ID from prior knowledge) — [arXiv](https://arxiv.org/abs/2002.08676)
- Mandi et al. (JAIR 80, 2024): gradient-based vs gradient-free DFL, 11 methods on 7 problems. Positions itself against Mišić and Perakis 2020 (applications), Kotary et al. 2021 (differentiable optimization) and Qi and Shen 2022 (OR tutorial). Builds on PyEPO's common code base. — [arXiv 2307.13565](https://arxiv.org/html/2307.13565v4)
- ML4CO-Kit's PredictiveCO-Benchmark targets DFL on predictive CO problems. — [ML4CO-Kit docs](https://ml4co-kit.readthedocs.io)

### Inferences
- DFL has its own benchmarks (PredOpt, PyEPO, PredictiveCO) separate from the neural-solver benchmarks. Cross-citation between the two communities seems limited.

### Gaps
- Not covered: 2025-2026 DFL advances (e.g., surrogate losses at scale, DFL with LLMs) and how widely CvxpyLayers/PyEPO are maintained and used in 2026.
