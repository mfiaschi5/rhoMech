# Batched GPU QM: go/no-go spike (2026-11-22)

Question: can a batched GPU path make `ksmech-qm run` (GFN2-xTB / ALPB-water, the geometry
optimization plus the final single point and the Fukui N+-1 points) at least 2x faster than the
production CPU path with 12 workers, at 30-80 atoms?

Hardware: RTX 4080 SUPER (16 GB; FP64 at 1/64 of the FP32 rate), Ryzen 7 2700X (8 cores,
16 threads). The CPU runs below are on one core with OpenBLAS preloaded (as the QM workers do,
`qm.blas`), on a machine running a 12-worker job (load ~15). That is close to one worker's speed
in a 12-worker production run on this machine. "cpu12" = one core's time / 12.

Molecules: 64 cached closed-shell records per bin (PMechDB + ReactMech, total atoms with H),
`sample.py`. Scripts and raw results: `data/processed/gpu_qm_spike/` (local, gitignored).

## Where the time is

Median seconds per record, from the QM manifests (161k records):

| atoms | <=15 | 16-30 | 31-50 | 51-80 | 81-120 | >120 |
|---|---|---|---|---|---|---|
| ReactMech, optimized | 0.4 | 2.4 | 11.8 | 46 | 221 | 432 |
| PMechDB, optimized | 0.3 | 1.6 | 7.4 | 29 | 157 | 661 |
| QMugs `given` (3 single points + sterics) | 0.3 | 0.8 | 1.5 | 2.5 | 5.0 | 12.4 |

- The optimizer is about 90-95% of an optimized record's time above 30 atoms.
- About 90% of ReactMech's 197 CPU-hours lie at 51 atoms and more: 70 h at 51-80, 97 h at
  81-120.

## Measurements

**1. tblite single point + gradient, ALPB-water, one core** (`cpu_tblite.py`). With OpenBLAS
it is 1.0-2.3x faster than the wheel's netlib BLAS.

| atoms | <=15 | 16-30 | 31-50 | 51-80 | 81-120 | >120 |
|---|---|---|---|---|---|---|
| ms, fresh SCF | 5 | 33 | 91 | 254 | 919 | 1431 |
| cpu12 ms | 0.4 | 2.8 | 7.6 | 21 | 77 | 119 |
| restarted / fresh (same geometry, 1-2 cycles) | 0.53 | 0.44 | 0.48 | 0.38 | pending | - |

A restarted single point still costs 0.38-0.53 of a fresh one. So about half of tblite's time
is integrals, gradient, ALPB and D4, not the SCF cycles. A GPU engine has to speed up that half
too.

**2. dxtb (main, 6961f9c, pure-torch integrals), GFN2 gas, batched on CUDA, fp64**
(`gpu_dxtb.py`).

| atoms | <=15 | 16-30 | 31-50 | 51-80 | 81-120 | >120 |
|---|---|---|---|---|---|---|
| ms per molecule, batch 16 | 5218 | 1754 | 3091 | 3499 | OOM | OOM |
| ms per molecule, batch 64 | 2617 | 2328 | OOM | OOM | OOM | OOM |
| max abs(E - tblite gas), Eh | 1e-6 | 2e-6 | 3e-6 | 2e-5 | - | - |

- It is 100-6000x slower than cpu12.
- The GPU sat at about 1% use.
- Breakdown at 31-50 atoms, batch 16 (`dxtb_timer.py`): of 75 s, the integrals take 32 s
  (Python loops over shells), the SCF 5 s, and the autograd backward most of the rest.
- **dxtb is out**, both as an engine and as a dependency.

**3. One batched SCF cycle on the GPU, lower bound** (`gpu_kernel.py`). The cycle is
X^T F X, eigh, back-transform, density, Mulliken charges and the Fock update with 9 multipole
contractions. Matrices are padded to the bin's largest AO count, best batch of 64/256/1024.

| atoms | <=15 | 16-30 | 31-50 | 51-80 | 81-120 | >120 |
|---|---|---|---|---|---|---|
| AO count, largest | 74 | 100 | 153 | 230 | 309 | 454 |
| fp64 ms per molecule-cycle | 0.083 | 0.134 | 0.293 | 0.680 | 1.28 | 3.31 |
| fp32 ms per molecule-cycle | 0.015 | 0.023 | 0.057 | 0.102 | 0.346 | 0.726 |

**4. fp32 eigensolver on the records' real Fock matrices** (`fp32_check.py`, 8 per bin). S^-1/2
is built in fp64 and only the eigh runs in fp32. Against fp64:
- AO charges agree within 2.1e-6;
- the band energy agrees within 7e-6 Eh.

This is accurate enough for an optimizer converging to fmax 3e-4.

**5. Polish share** (`checks_cpu.py`, 4 molecules per bin, RDKit start). This is the share of
an optimization's energies spent after max abs(g) first drops below 1e-3:

| atoms | <=15 | 16-30 | 31-50 | 51-80 | 81-120 |
|---|---|---|---|---|---|
| L-BFGS, median (range) | 0.50 (0.23-0.60) | 0.56 (0.23-0.72) | 0.59 (0.33-0.68) | 0.58 (0.35-0.80) | pending |
| ANC, median (range) | 0.33 (0.25-0.50) | 0.46 (0.31-0.93) | 0.69 (0.63-0.82) | 0.58 (0.18-0.61) | pending |

ANC also needed about half of L-BFGS's energies here, as in the benchmark of 2026-11-09.

## Verdict

**No-go:**
- **dxtb**, by measurement 2.
- **An all-fp64 GPU engine.** SCF cycles alone at 15 cycles per SCF, with the rest costing as
  much again, project to 0.9-1.9x against cpu12 at 31-80 atoms. That is below the 2x gate.
- **The plan's split** (GPU optimizer down to fmax 1e-3, tblite polish down to 3e-4 on the
  CPU). About half of the energies come after 1e-3 (measurement 5), so the CPU would keep
  about half the work. That caps the end-to-end gain near 2x however fast the GPU is.

**Not yet decided: an engine of our own that optimizes on the GPU all the way to fmax 3e-4.**
It would assemble energy and gradient in fp64 with the eigensolver in fp32 (measurement 4).
tblite would do only the record's final single point (already in the pipeline) and the Fukui
points, with a CPU polish only where tblite's gradient there exceeds the limit. The GPU must
fit integrals, their gradients, D4, ALPB and repulsion within these budgets per molecule and
energy to reach 2x:

| atoms | <=15 | 16-30 | 31-50 | 51-80 | 81-120 | >120 |
|---|---|---|---|---|---|---|
| budget for the non-SCF part, ms | 0.06 | 0.8 | 2.3 | 6.8 | 25 | 37 |

The budget is cpu12 x 0.7 (a restarted SCF in an optimizer) / 2, minus 6 fp32 cycles. A flop
count of the GFN2 overlap and multipole integrals with gradients (about 4e8 flop per molecule at
51-80 atoms) suggests a few ms. That makes 51 atoms and up plausible (about 90% of the CPU-hours),
31-50 tight, and 30 atoms and below out of reach. This is an estimate, not a measurement.

**What decides it.** Write the engine's first piece, the batched overlap, dipole and quadrupole
integrals of the GFN2 basis with gradients (about a day), and time it against the budgets. It is
the engine's step 1 either way, so nothing is lost if the answer is go.

**The CPU alternative, for comparison.** `--optimizer tblite-anc` already gives about 2x fewer
energies. It also needs a full recompute, just like a GPU method tag.

## Stage 1 result (2026-11-22)

**The integrals gate passes.** The batched fp64 CUDA kernels (`native/ksxtb`) compute S, H0,
D and Q with tblite's conventions, plus the H0 gradient contraction. They agree with tblite
on 96 of these molecules to machine precision: S 1e-15, H0 2e-13, and dipole and quadrupole
4e-13.

At 51-80 atoms one forward plus gradient pass costs 0.14 ms per molecule, against a gate of
2.27 ms and 21 ms for cpu12's whole single point.

Every bin passes. The table and the design are in docs/decisions.md, 2026-11-22. The flop
estimate above ("a few ms") was pessimistic by about 20x.

## Stage 2 result (2026-11-22)

**Every other GFN2 term passes too.** Repulsion, D4, ES2, ES3, AES, ALPB and CDS run batched
on the GPU in fp64 (`native/ksxtb/src/classical.cu`), with their potentials and gradients.
With stage 1's integrals, the whole energy and gradient at tblite's converged density match
tblite on 48 spike molecules: energy 4e-12 Eh, gradient 2.6e-11 Eh/Bohr, ALPB water.

All the work outside the SCF of one energy and gradient costs 0.49 ms per molecule at 51-80
atoms and 2.5 ms above 120, 5-10% of the budget from 16 atoms up. Only the smallest bin is
tight (0.026 of 0.040 ms). The table is in docs/decisions.md, 2026-11-22. What remains is the
SCF: the eigensolver and one potential evaluation per cycle (0.16 ms at 51-80 atoms).

## Stage 3 result (2026-11-22)

**The batched SCF single point beats cpu12 in every bin.** `ksmech.qm.gpu.scf` runs tblite's
SCF for a batch: SAD guess, Broyden mixer, convergence criteria. Cycles run in fp32 and finish in
fp64. With both converged hard it lands on tblite's fixed point (energy 4e-12 Eh, gradient
9e-11 Eh/Bohr on 48 spike molecules). At production accuracy it lands where tblite does.

A fresh single point with gradient takes 3.9 ms per molecule at 51-80 atoms, against cpu12's
21 ms (5.4x; the gate was <= 10.5 ms), and 0.7-25 ms from 16 to 200 atoms (3.9-7.8x). Below
16 atoms it is about even (1.1x). A restarted SCF, as an optimizer runs it, is 20-30% cheaper.
Building the batch on the host is outside these times: 2.5 ms per molecule at 51-80 atoms, so a
lone single point there is 6.4 ms (3.3x); an optimizer pays it once per record.
The table is in docs/decisions.md, 2026-11-22. The end-to-end question, a whole optimized record
against `ksmech-qm run --workers 12`, is stage 4's (the batched optimizer).

## Stage 4 result (2026-11-23)

**The batched optimizer reproduces production's.** `ksmech.qm.gpu.optimize.relax_many` runs
production's tblite-anc `_relax` (free run with the dissociation guard, walled retry) for many
molecules at once on the GPU. On 96 spike species it gives the same outcome for every one, the
same walled retries (15), the same energy counts (median ratio 1.0), and end points tblite
accepts (79 of 81 free runs below fmax, the other two above by 1e-6). 69 land in the same
minimum, most of the rest in the same flat basin.

Throughput against production's CPU speed (12 workers with the OpenBLAS preload, estimated
from the reference): even (1.17x) at 51-80 atoms with 256 molecules in flight, slower with
fewer, 3.1-3.3x above 80 atoms, and much slower below 30 atoms. The spike's 2x bar at 30-80
atoms is missed; the SCF's per-step cost is the lever. Details in docs/decisions.md,
2026-11-23; stage 6 measures it end to end.

## SCF tuning result (2026-11-23)

**The 2x bar at 30-80 atoms is not met: about 2x at 51-80 atoms on a long queue, 1.5x at
31-50.** Changes:
- one fp32 eigensolve per SCF cycle, with the precise phase's occupied orbitals refined in fp64
  instead of an fp64 eigensolve (fixed cost of 4 ms per call);
- a Cholesky mixer with preallocated history;
- kernel layouts that keep small batches spread over the device.

Against 12 CPU workers on the same species (optimization alone), 1024 conformers in flight:
2.06-2.07x at 51-80 atoms (0.33 against 0.682 s per molecule), 1.47-1.51x at 31-50. With 256
conformers the end of the set drains the GPU: 1.45x and 1.1x. Above 80 atoms: 3.1x and 3.3x,
2-3% faster than stage 4.

Parity gate: 96/96 outcomes. One 127-atom Pd complex took a walled retry the CPU did not need;
alone it does so in 1 of 4 runs. The other 15 walled retries match.

What is left is the SCF straggler (a step runs as many cycles as its slowest molecule) and
each cycle's fixed cost. Details in docs/decisions.md, 2026-11-23.

## SCF carry-over result (2026-11-24)

**The 2x bar is met at 51-80 atoms on a long queue (2.26x), but not at 31-50 (1.79x).**
A molecule whose SCF is slow now keeps its SCF state into the next optimizer round, while the
others take their steps. Before, each step waited for the slowest molecule's SCF. The SCF has
per-molecule state and mixer (`SCFBatch.new_run/start/advance/result`). The optimizer runs in
rounds (`BatchedANC(carry=...)`). `relax_many`'s default is carry 0.1: a round ends once at
most a tenth of its SCFs still run.

Against 12 CPU workers on the same species (optimization alone), 1024 conformers in flight:
- 0.120 s per molecule at 31-50 atoms (CPU 0.214);
- 0.302 at 51-80 (CPU 0.682).

That is 8% and 6.5% faster than the same code without carry, and 16% and 9% faster than the
SCF-tuning build. With 256 conformers carry gains nothing; above 80 atoms (3.2x) it costs 2.5-4%: the
batches are small.

Parity gate: 96/96 outcomes. One 127-atom Pd complex takes a walled retry the CPU does not.
With carry it does so in every run (5 of 5), without carry in none (0 of 4). It still ends ok,
at a minimum 2.8 mEh lower. Details in docs/decisions.md, 2026-11-23 and 2026-11-24.

## Stage 5 result (2026-11-25)

**`ksmech-qm run --optimizer gpu-anc` works end to end.** The GPU optimizes the untagged closed
shells of 31 atoms and more, and the CPU workers do everything else. Every record is stored
under `GFN2-xTB/ALPB-water/gpu-anc`.
- **Finishing:** the workers write each record from tblite's single point at the GPU
  geometry, with a short tblite-anc polish where tblite's gradient there is above fmax.
- **Failures:** a GPU failure is rerun on the CPU, and only the CPU's outcome reaches the
  failure memo.

Smoke runs on the spike's 384 species (1-200 atoms):
- 384/384 records in each of the three runs: 256 optimized on the GPU, 3-5 of them polished,
  none retried on the CPU.
- 467-503 s with 12 workers. An estimate for the CPU alone on this sample is about 810 s,
  about 1.7x, dominated by the species above 80 atoms; stage 6 measures it end to end.
- The GPU records' energies match production's tblite-anc from the same starts as closely as
  the parity gate did.

Details in docs/decisions.md, 2026-11-25.
