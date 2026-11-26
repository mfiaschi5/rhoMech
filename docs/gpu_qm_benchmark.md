# GPU xTB stage 6: end-to-end benchmark of `ksmech-qm run` (2026-11-26)

**Question:** how fast `ksmech-qm run --optimizer gpu-anc` (the batched CUDA GFN2-xTB/ALPB-water
optimizer of `ksmech.qm.gpu`, with its CPU tail) makes records, end to end, against the CPU
alone with 12 workers. The CPU baseline is measured twice, once with production's optimizer
(`tblite`, the L-BFGS-B method `.../tblite-lbfgs`) and once with `tblite-anc`. The two CPU arms
separate the GPU's gain from the optimizer's.

**Answer:**
- On 1,000 random ReactMech species, gpu-anc makes records **1.55-1.70x faster than
  tblite-anc** and **5.1-5.7x faster than production's tblite-lbfgs**. Of the second figure,
  3.3x is the optimizer: tblite-anc on the CPU alone is 3.3x faster than tblite-lbfgs.
- The GPU's own gain (1.6x) is below the spike's 2x bar. The run is GPU-bound: after the
  first minutes the 12 CPU workers are 70-85% idle.
- The outcomes are the same. On the 758 species the GPU optimized, 77% of gpu-anc's records
  lie within 1e-5 Eh of tblite-anc's and 99% within 1e-3. tblite-lbfgs's records lie within
  1e-5 of tblite-anc's for only 5%.
- gpu-anc does not repeat itself bit for bit: its two runs differ beyond 1e-4 Eh on 25 of
  those species.

## Setup

- **Hardware:** RTX 4080 SUPER (16 GB, the full-rate unlock of 2026-11-22 installed) and a
  Ryzen 7 2700X (8 cores, 16 threads, 32 GB).
- **Code:** branch `claude/gpu-xtb` at 83b1e77, clean tree. Every arm runs
  `ksmech-qm run --species-from FILE --workers 12 --timeout 900 --min-success 0` with
  `OMP_NUM_THREADS=1` and the default BLAS (OpenBLAS in the workers). It changes only
  `--optimizer` (`tblite`, `tblite-anc`, `gpu-anc`).
- **Fresh state:** each (set, arm) gets a fresh `--cache-dir` and `--processed-dir`. Every ok
  record is checked to have been written during its own arm, and there are no memo hits.
- **Wall time:** the whole command, from Python start to the manifest: census, embedding,
  optimization, single points, Fukui and sterics.
- **One queue job** (`data/gpu_queue`, job 50, 01:50-04:22) held the GPU and kept fits off the
  machine. Before each arm the job waited for 85% CPU idle over 5 s and for no pytest or
  ksmech fit running.
- **Monitoring:** `vmstat` every 5 s, `nvidia-smi` every 1 s, and every 30 s a scan for foreign
  pytest or fit processes and for other CPU-heavy processes.
- **Contamination:** none seen. No pytest or fit ran during an arm. The CPU arms on the slice
  averaged 75.7% CPU in `vmstat`, which is the 12 workers on 16 threads plus less than one
  thread for everything else (the desktop included).
  - The 30 s scan of other busy processes is weak evidence. It reads lifetime-average %CPU,
    leaves out the browser and the desktop, and logged the arms' own workers: `timeout` puts
    them in their own process group. Outside those it found one defunct process, in the last
    seconds of the last arm.
- **Inputs** (`data/processed/gpu_qm_benchmark/`, local; scripts in `scripts/gpu_xtb/stage6_*`):
  - `reactmech_1000`: 1,000 species drawn at random (seed 0) from the full ReactMech v19 QM
    manifest, any status, the spike's species excluded. By total atoms: 26 up to 15, 211 at
    16-30, 346 at 31-50, 292 at 51-80, 114 at 81-120, 11 at 121-200.
  - `spike_<bin>`: the spike's 64 species per size bin (`gpu_qm_spike/sample.py`). They were
    drawn from records that tblite-lbfgs had converged, which flatters lbfgs's converged
    fraction there.
- **Repeats:** one run per arm, except gpu-anc on the ReactMech slice, which ran twice for a
  noise figure.

## The ReactMech slice (the realistic mix)

| Arm | Wall s | Species/h | ok | Failures |
|---|---|---|---|---|
| tblite-lbfgs | 2892 | 1,245 | 983 | OPT_NOT_CONVERGED 11, SCF_NOT_CONVERGED 2, TIMEOUT 1, SPIN_CENSUS_REFUSED 3 |
| tblite-anc | 871 | 4,133 | 994 | OPT_NOT_CONVERGED 3, SPIN_CENSUS_REFUSED 3 |
| gpu-anc, run 1 | 511 | 7,046 | 994 | the same as tblite-anc |
| gpu-anc, run 2 | 562 | 6,400 | 994 | the same as tblite-anc |

- **Speed-ups:**
  - gpu-anc / tblite-anc: 1.70x and 1.55x;
  - gpu-anc / tblite-lbfgs: 5.66x and 5.14x;
  - tblite-anc / tblite-lbfgs: 3.32x.
- **Noise:** the two identical gpu-anc runs differ by 10%, so a single run resolves nothing
  finer than that.
- **gpu-anc routes:**
  - 761 species optimized on the GPU, 235 small ones (below 31 atoms) and 1 open shell on
    the CPU; the 3 SPIN_CENSUS_REFUSED stop before routing.
  - 758 GPU outcomes were ok.
  - 3 were retried on the CPU: the slice's 3 OPT_NOT_CONVERGED, which failed again there,
    as in tblite-anc.
  - 7 and 11 of the GPU geometries got a tblite-anc polish (GPU_POLISHED).
- **Where the time goes in gpu-anc:**
  - In the first quarter of the run the CPU is 68% busy (the small species, embedding) and the
    GPU 34-39%.
  - After that the CPU is 15-29% busy and the GPU 64-82%: the GPU sets the pace.
  - GPU means over the run: 62% use, 73 W (peak 131 W of the 320 W limit), peak memory
    6.3-8.0 GB.
- **A whole source is not measured.** These rates hold for a 1,000-species queue. The GPU ran
  at 34-39% in the first quarter, and stage 4 showed its gain at 31-80 atoms growing with the
  queue's length, so a longer queue may do better; how much is unmeasured. For scale only:
  ReactMech's 129,655 species at these rates would be about 104 h with tblite-lbfgs, 31 h with
  tblite-anc and 18-20 h with gpu-anc. That is a linear scaling, not a prediction.
- **Production's own timings are slower.** For the same species, the production manifest's
  seconds are 1.7x the lbfgs arm's (median per species). The ratio grows with size: 0.94x at
  1-15 atoms, 1.6x at 31-50, 2.0x at 51-80, 2.3x at 81-120, 2.5x above 120. 293 of 300 sampled
  production records were written on 2026-10-26 to 11-09, before the workers ran tblite on
  OpenBLAS (ba36049, 2026-11-09). The spike measured OpenBLAS at 1.0-2.3x faster, growing with
  size, so the preload is the likely cause; other load is not ruled out.

### Record differences (common ok species)

The gpu-anc rows cover only the species the GPU optimized. Below 31 atoms gpu-anc runs
tblite-anc on the CPU, and those 235 records are bit-identical to tblite-anc's (dE 0, RMSD
1e-15).

| Comparison | n | Median abs dE, Eh | p90 | < 1e-5 | < 1e-4 | < 1e-3 | Max | Median RMSD, Å | p90 RMSD | RMSD < 0.1 Å |
|---|---|---|---|---|---|---|---|---|---|---|
| gpu-anc run 1 vs tblite-anc | 758 | 8.7e-7 | 5.3e-5 | 584 | 705 | 749 | 1.9e-2 | 0.002 | 0.051 | 711 |
| gpu-anc run 2 vs tblite-anc | 758 | 6.4e-7 | 5.2e-5 | 594 | 700 | 748 | 7.3e-3 | 0.002 | 0.050 | 705 |
| gpu-anc run 1 vs run 2 | 758 | 9.3e-9 | 1.6e-5 | 661 | 733 | 752 | 1.9e-2 | 0.00003 | 0.017 | 732 |
| tblite-lbfgs vs tblite-anc (all sizes) | 980 | 2.8e-4 | 2.2e-3 | 50 | 250 | 785 | 3.3e-2 | 0.16 | 0.71 | 338 |

- RMSD is the Kabsch RMSD of all atoms, in the records' common atom order.
- **tblite-anc has no run-to-run noise here** (the CPU-routed records came out bit-identical).
  The spread between gpu-anc and tblite-anc is the GPU's: batch composition changes a
  molecule's path, so two gpu-anc runs differ beyond 1e-4 Eh on 25 of 758 species.
- **Sign of the tail:** of gpu-anc's records more than 1e-4 Eh from tblite-anc's, 33 of 53 end
  lower (run 1; run 2: 33 of 58). The largest is 1.9e-2 Eh lower in run 1. There is no bias
  toward worse minima.
- **By size bin**, the share of gpu-anc records within 1e-5 Eh of tblite-anc falls with size:
  87% at 31-50 atoms, 71% at 51-80, 65% at 81-120.
- **The rest** are nearby minima of a flat surface: 99% within 1e-3 Eh, and a median RMSD
  under 0.01 Å in every bin.
- **Scale:** switching the optimizer from L-BFGS-B to ANC moves records far more than moving
  ANC from the CPU to the GPU: 75x at the 90th percentile of dE and 24x in RMSD, and by
  orders of magnitude at the median.

## Per size bin (64 species each: a short queue)

| Atoms | tblite-lbfgs s | tblite-anc s | gpu-anc s | gpu / anc | gpu / lbfgs | anc / lbfgs | GPU use |
|---|---|---|---|---|---|---|---|
| 1-15 | 5.8 | 4.6 | 8.5 | 0.54x | 0.68x | 1.26x | none (all CPU) |
| 16-30 | 20.8 | 14.8 | 12.2 | 1.21x | 1.71x | 1.41x | none (all CPU) |
| 31-50 | 66.9 | 24.3 | 31.6 | 0.77x | 2.12x | 2.75x | 33% |
| 51-80 | 234.1 | 63.7 | 68.0 | 0.94x | 3.44x | 3.68x | 47% |
| 81-120 | 742.4 | 238.4 | 175.5 | 1.36x | 4.23x | 3.11x | 60% |
| 121-200 | 1520.3 | 446.6 | 241.3 | 1.85x | 6.30x | 3.40x | 60% |

- **Outcomes:** tblite-anc and gpu-anc made 64/64 records in every bin. tblite-lbfgs failed 1
  species at 51-80, 1 at 81-120 and 2 at 121-200, all OPT_NOT_CONVERGED. gpu-anc polished 1
  species at 31-50 and 2 at 121-200.
- **Below 31 atoms both gpu-anc runs are tblite-anc relabelled**, an A/A control.
  - At 1-15 atoms they show the arm's fixed cost of about 4 s (CUDA start-up).
  - At 16-30 they show the noise of a 15 s run: the "1.21x" there is noise.
- **At 31-80 atoms a 64-species queue is too short for the GPU.** Its batches stay small, as
  stage 4 found (1.44x with 256 conformers in flight, 2.26x with 1,024). The GPU wins only
  above 80 atoms here. These are short-queue figures and are not to be extrapolated to a
  whole source: the slice above is the realistic mix.

## Reading

1. **The GPU gives 1.6x over the same optimizer on the CPU**, on a realistic slice of 1,000
   species. That is below the 2x bar the spike set at 30-80 atoms, and short queues there are
   a loss. The gain comes from the large species and from long queues. A whole source, a much
   longer queue, is unmeasured.
2. **tblite-anc alone is 3.3x over production's optimizer** in wall time (3.6x in summed
   worker seconds), as the 2026-11-09 benchmark found (3.1x on 34 molecules).
   - The gain is spread over all species. lbfgs also fails more (17 against 6), but its
     failures take only 9% of its worker seconds.
   - Either switch is a new method tag and a full QM recompute (`docs/decisions.md`,
     2026-11-09).
   - gpu-anc's further gain over tblite-anc is 1.6x on these 1,000 species.
3. **The gpu-anc pipeline is GPU-bound, with the CPU mostly idle.** Levers, none built or
   measured:
   - CPU workers that also take GPU-eligible species while they wait (work sharing). If the
     ~75% idle share of the 12 workers ran tblite-anc at its measured rate, the rates would add
     to at most about 2.3x over tblite-anc (1.62 + 0.75). That is an upper estimate.
   - A size-dependent threshold that leaves 31-50 atoms on the CPU.
   - A smaller one: the 3 GPU failures on the slice were the 3 OPT_NOT_CONVERGED, which the CPU
     retry repeated and failed the same way (13-61 worker seconds each, on idle workers).

Raw data: `data/processed/gpu_qm_benchmark/runs/<set>__<arm>/` holds `wall.json`, `run.log`,
the manifest, the records, `vmstat.txt`, `gpu.csv`, `state.txt`, `foreign.txt` and
`busy.txt`. `compare.txt` and `bench_summary.json` hold the comparison.
