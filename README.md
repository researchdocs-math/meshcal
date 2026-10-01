# Meshcal

Rank-directed conformal calibration with deterministic finite-element response enclosures. The experiments quantify how discretization error in calibration labels changes continuum-response coverage and how selective numerical refinement controls calibration precision.

The package contains the complete simulation inputs, finite-element response database, frozen surrogate predictions and neural weights, numerical comparison results, solver implementations, and verification tests. All three authors contributed equally: Cemil Emre Yavas, Nancy R. Mead, and Lei Chen.

## Quick reproduction

Use Python 3.12 in a virtual environment. From this directory:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python -m meshcal.reproduce --mode replay
```

The replay verifies frozen predictions, repeats all calibration policies and sensitivity analyses, and checks the new numerical results against `results/metrics.csv`. Outputs are written to `results-reproduced/`. The response database makes this route practical in minutes without repeating the finite-element solves. Exact reproduction uses the supplied fixed predictions; inference reconstructed from fitted GP parameters agrees within floating-point tolerance.

## Regenerate the finite-element experiments

```sh
python -m meshcal.reproduce --mode full \
  --data data-regenerated --out results-regenerated --workers 12
```

This draws the declared seeded inputs, solves every requested uniform and higher-order reference problem, and repeats the comparisons. Completed response files are checked against their input arrays and reused when restarting. The original experiment used an Apple M3 Pro with 12 CPU cores, 18 GPU cores, and 18 GB memory. Twelve independent CPU processes perform assembly and sparse solves; each process uses one linear-algebra thread. Adjust `--workers` for the available memory and CPU cores. Runtime depends on the machine, solver libraries, and concurrent workload. Timing columns are measurements and will change on another execution.

For fresh-solve policy verification:

```sh
python -m meshcal.live --workers 12 --replicates 2
```

This runs Rank, Uniform, and direct Reference-bounds calibration on the first two calibration replicates, for both GP predictors and all geometries. It checks the thresholds and requested work against the stored comparisons. Its timings include assembly and linear solves after construction of the reusable finite-element spaces, with independent runs sharing the machine.

## Experimental design

- Three diffusion domains: unit square, L-shaped square, and square with a central square hole. Homogeneous Dirichlet conditions apply to every boundary.
- Six independent inputs uniform on `[-1.3, 1.3]`: four log conductivities and two log load amplitudes. The L-shaped domain has five active inputs.
- Per domain: 512 training inputs, 30 independent calibration sets of 199 inputs, and 2,000 independent shared test inputs. Seeds are 720026, 720027, and 720028.
- Uniform triangular meshes with edge subdivisions 8, 16, 32, 64, and 128. Primal P1, RT0 flux, and P0 multiplier spaces supply response bounds.
- A terminal P2 / first-order Raviart--Thomas / discontinuous P1 enclosure uses a graded mesh based on the 32-edge grid. The grading counts are 4, 12, and 4 for square, L-shaped, and perforated domains.
- Fixed coarse GP, midpoint GP, and four-member neural ensemble predictors. The midpoint training targets are log-bound midpoints on the 32-edge mesh.
- Main nominal coverage 90%, quantile precision 0.005. Sensitivity: nominal coverage 80%, 90%, and 95%; precision 0.002, 0.005, 0.01, and 0.02.

The statistical guarantee assumes exchangeable continuum labels and valid response enclosures. The floating-point realization uses a residual correction and an outward numerical guard. Numerical test-reference uncertainty is reported as a coverage bracket; it is distinct from variation across calibration samples. The 30 calibration replicates share a training and test set.

## Data and result formats

`data/<case>.npz` contains the training, calibration, and test input arrays and uniform-mesh responses. Calibration inputs have shape `(30, 199, 6)`. Response columns are lower energy, uncorrected upper complementary energy, elapsed seconds, primal unknown count, flux unknown count, triangle count, and relative equilibrium residual. `calibration.log_enclosures` applies the residual correction and outward guard.

`data/<case>_reference.npz` contains calibration and test responses on the terminal mesh. Its columns are corrected lower bound, corrected upper bound, elapsed seconds, triangle count, total unknown count, and the L2 equilibrium residual. Individual split-and-level archives retain the same inputs for restartable generation. Metadata are in `generation.json` and `reference_generation.json`.

`results/<case>_metrics.csv` and `results/metrics.csv` record every method and replicate. Columns include the upper score quantile, lower/upper empirical continuum coverage, numerical-training-label coverage, relative interval width, requested unknown-count work, recorded solve time, achieved quantile gap, convergence flag, and number of requests. The Naive baseline has no certified-work comparison; its work and timing entries are empty. `Reference midpoint` is a point-label comparator; its convergence flag does not assert a numerical precision certificate.

`results/<case>_sensitivity.csv` records coverage/precision sweeps. `*_h_only.csv` retains the uniform-mesh-cap ablation. `*_allocations.npz` records final fidelity indices (0 through 5); `*_traces.npz` records work, lower quantile, upper quantile, and request count for the first five replicates. Diagnostic JSON files record surrogate errors, reference widths, and the environment. Work counts sum all requested primal, flux, and multiplier unknowns, including repeated refinements. It is a deterministic discretization measure, not a floating-point-operation estimate.

## Surrogate training

The archived predictors fix the experiment independently of optimization-library differences. To repeat GP optimization:

```sh
python -m meshcal.train_gp --out results-refitted
```

Optional neural retraining uses PyTorch and automatically selects Metal when available:

```sh
python -m pip install -r requirements-neural.txt
python -m meshcal.neural --case square --out results-refitted
python -m meshcal.neural --case notch --out results-refitted
python -m meshcal.neural --case hole --out results-refitted
```

Neural training uses four 6-64-64-1 tanh networks, 6,000 full-batch Adam steps, a fixed seed of 9172, and cosine learning-rate decay. Device-dependent training may produce slightly different weights. Stored weights are evaluated with portable double-precision NumPy inference. Retraining is distinct from reproducing the published fixed-predictor comparisons.

## Software interface

`refine_query` in `meshcal.calibration` accepts a fixed predictor array, known request costs, and a callback returning a log-response enclosure for an input and fidelity index. It implements rank, width, random, and uniform allocation. `refine` uses the same interface for a response database. Independent solvers can implement the callback without changing calibration logic.

## Verification and license

The test suite checks analytical compliance enclosures, manufactured-solution convergence, nested energy bounds, adaptive quantile domination, precision certificates, ties, invalid enclosures, non-unit scales, and small-sample behavior. `SHA256SUMS` records the archived file checksums. Source code and experiment data are released under the MIT license in `LICENSE`; author metadata are in `CITATION.cff`.
