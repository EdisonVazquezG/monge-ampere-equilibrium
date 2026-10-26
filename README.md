# Convex equilibrium potentials for Monge–Ampère

Code for

> E. Vázquez, Y. Lu, T. Nowicki. *Convex Equilibrium Potentials for Monge–Ampère: Curvature Bounds and A Posteriori Evaluation.*

The networks are nonnegative softplus equilibrium models

$$z^*(x)=\phi(Wz^*(x)+Ax+b),\qquad u_\theta(x)=w^\top z^*(x)+a^\top x+c,\qquad W\ge0,\ w\ge0,\ \|W\|_\infty\le q<1,$$

trained on the Dirichlet problem $\det D^2u=f$ in the unit disk, $u=g$ on its boundary. Trained models are certified with Arb ball arithmetic (128 bits): pointwise derivative and determinant bounds at approximate equilibria, and uniform residual and boundary bounds from complete covers of the disk and the circle.

**Paper:** arXiv:2610.31386 (main article and supplement). **Archived code and complete results:** Zenodo DOI `10.5281/zenodo.XXXXXXX` (add after the first release).

## What this repository shows

- Certified trained models: uniform residual and boundary bounds from complete covers of the disk and circle, computed with Arb.
- Honest comparisons, not a leaderboard: on these smooth two-dimensional problems an explicit ICNN of the same size is cheaper and often as accurate, and a finite-difference solver is far more accurate. The distinguishing property of the equilibrium class is the parameter-only, depth-independent curvature bound that follows from the contraction constraint.

## Quick start

Open `monge_ampere_experiments.ipynb` (locally or in Google Colab, CPU runtime) and run all cells. Set `PROFILE = "smoke"` for a check of the whole chain in a few minutes, or `PROFILE = "full"` for the actual protocols.

From a terminal:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd code
python test_certification.py                                   # certificate unit tests
python validation_e0_e1.py --out ../results/validation_e0_e1    # E0-E1 derivative validation
python run.py --profile full --out ../results/paper_full        # paper experiments (main article, numerical section)
python refine_saved.py --source-zip ../results/paper_full_results.zip --out ../results/refinement_P2_seed2
python extended.py --profile full --out ../results/extended_full --paper-run ../results/paper_full
```

## Contents

| File | Purpose |
|---|---|
| `monge_ampere_experiments.ipynb` | Runs everything below in order and displays tables and figures |
| `code/numerics.py` | Float64 evaluation, residual estimator of Theorem 5.1, adaptive stopping |
| `code/rigorous.py` | Arb certificates for points and boxes (Proposition 6.1), verified implicit solves |
| `code/coverage_check.py` | Exact combinatorial audit of disk and boundary covers |
| `code/original_pilots.py` | Network, training protocol and backward checks of the original pilots |
| `code/run.py` | Paper experiments: training, depth sweep, Arb point audit, timing, covers, report |
| `code/refine_saved.py` | Bounded refinement of the P2 seed-2 cover and barrier bounds (refinement figure and table) |
| `code/check_bounds.py` | 120-network float64 consistency experiment (supplementary consistency table) |
| `code/test_certification.py` | Unit tests of the certificates (80-digit oracle, exact cases, coverage gaps) |
| `code/validation_e0_e1.py` | Implicit derivatives vs autograd; a priori and residual state bounds |
| `code/extended.py` | Extended experiments X1–X6 (below) |
| `code/icnn_certification.py` | X7: the explicit ICNN certified with the same Arb machinery, compared with the equilibrium models |
| `code/config_full.json`, `code/config_extended.json` | All protocol settings |
| `code/METHODS.txt` | What exactly is certified, and how rounding is handled |
| `history/e3_exploratory_notebook.ipynb` | Earlier exploratory notebook (not the source of the paper's numbers) |

## Experiments

**Paper (main article, numerical section).** P1: $u^*=x_1^2+x_2^2/4$, $f=1$. P2: $u^*=\tfrac12\|x\|^2+0.1(x_1^4+x_2^4)$, $f=(1+1.2x_1^2)(1+1.2x_2^2)$. Three seeds each; width 12, $q=0.7$, 32 unrolled iterations; 1,200 Adam steps then up to 100 L-BFGS iterations; boundary weight 10; 2,048 interior and 512 boundary evaluation points.

**Extended (`extended.py`).**

| ID | Experiment |
|---|---|
| X1 | Near-critical contraction, $q\in\{0.7,0.9,0.95,0.98\}$ with $\|W\|_\infty=q$ exactly: accuracy, both branches of the curvature bound vs the observed Hessian, cancellation $S\gamma\le1$, adaptive depth and Arb cost |
| X2 | Explicit ICNN (3 hidden layers of width 8, 211 parameters) vs the equilibrium network (207 parameters), same data and optimizer protocol |
| X3 | Harder problems with the full certification pipeline. P3: $u^*=e^{\|x\|^2/2}$, $f=(1+\|x\|^2)e^{\|x\|^2}$ (radial exponential in the spirit of the examples of Nyström–Vestberg). P4: $u^*=x_1^4/12+x_2^2/2$, $f=x_1^2$ (degenerate on $x_1=0$) |
| X4 | Uniform residual bound $R$ versus the number of certified boxes and CPU time under priority refinement |
| X5 | Finite-difference reference: centred differences, Benamou–Froese–Oberman iteration and Newton polish; exterior stencil nodes take exact values |
| X6 | Cost of float64 evaluation and of Arb point and box certificates for widths $m\in\{12,24,48\}$ |
| X7 | `icnn_certification.py`: Arb certification of the explicit ICNNs of X2 (natural interval extension of the exact jet recursion) with the same boxes and priority refinement as X4: pointwise cost, uniform residual $R$ vs boxes and time, depth sweep (3, 6, 12 layers) and a plain a priori Hessian bound |

The certified modules are used unchanged by the extended experiments; new problems are registered at import time in `extended.py`, so the hashes recorded by `run.py` and checked by `refine_saved.py` remain valid.

## Results

Generated results (`results/`) are not stored in this repository. The complete outputs of the runs reported in the paper (`paper_full`, `refinement_P2_seed2`, `extended_full`, `X7_icnn_certification_full`) are archived on Zenodo; each run records `RUN_STATUS.json`, `protocol.json` and source hashes. Training is not bit-for-bit deterministic across machines: re-running reproduces the tables up to small differences in some trained models, while the saved certificates apply to the saved parameters.

Runtime on one CPU thread: the `full` profile takes several hours (the extended part is the longest); `smoke` takes minutes.

## Scope

The certificates are rigorous for the exact real model defined by the saved binary64 parameters; they do not certify the training process or its convergence. Sample maxima are not domain suprema; uniform statements come only from complete covers. Timing comparisons are made within cohorts of equal size and target. See `code/METHODS.txt`.

## Requirements

Python 3.11 or later, CPU only. PyTorch (CPU wheel), NumPy, Matplotlib, SciPy, mpmath, python-flint 0.8.0 and pandas.

## Citation

If you use this code, please cite the paper and the Zenodo archive (see `CITATION.cff` once added).

## License

MIT License. See `LICENSE`.
