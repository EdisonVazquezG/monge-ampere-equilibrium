#!/usr/bin/env python3
"""E0-E1 validation of the equilibrium evaluation (no training).

E0: implicit input derivatives (Proposition A.3) agree with autograd through an explicit
    weight-tied unroll, and with the exact derivatives when W = 0.
E1: state error against a tighter reference equilibrium decreases with depth and stays
    below the a priori and residual bounds.

Usage:  python validation_e0_e1.py --out results_e0_e1
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import argparse
import csv
import hashlib
import json
import platform
import shutil
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DTYPE = torch.float64


@dataclass
class Config:
    width: int = 8
    q: float = 0.7
    seeds: tuple = (0, 1, 2)
    n_points: int = 64
    n_derivative_points: int = 8
    state_atol: float = 1e-12
    reference_state_atol: float = 1e-14
    max_iter: int = 1000
    unroll_steps: int = 128
    check_atol: float = 1e-9
    check_rtol: float = 1e-7
    convexity_atol: float = 1e-10


def softplus(s):
    # logaddexp implementa log(1 + exp(s)) de forma estable, sin umbral lineal.
    return torch.logaddexp(torch.zeros_like(s), s)


def make_parameters(m, q, seed):
    """Untrained parameters; W >= 0 and every row sum < q."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.rand((m, m), generator=gen, dtype=DTYPE)
    row_budget = q * (0.5 + 0.5 * torch.rand((m, 1), generator=gen, dtype=DTYPE))
    W = raw / raw.sum(dim=1, keepdim=True) * row_budget
    return {
        "W": W,
        "A": 0.7 * torch.randn((m, 2), generator=gen, dtype=DTYPE),
        "b": 0.2 * torch.randn(m, generator=gen, dtype=DTYPE),
        "w": (0.2 + torch.rand(m, generator=gen, dtype=DTYPE)) / m,
        "a": 0.2 * torch.randn(2, generator=gen, dtype=DTYPE),
        "c": torch.tensor(0.0, dtype=DTYPE),
    }


def sample_disk(n, seed):
    """Area-uniform points: radius sqrt(U), uniform angle."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    radius = torch.sqrt(torch.rand(n, generator=gen, dtype=DTYPE))
    angle = 2 * np.pi * torch.rand(n, generator=gen, dtype=DTYPE)
    return torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)


def state_map(z, x, p):
    # Rows = samples; equivalent to W z + A x + b in column notation.
    return softplus(z @ p["W"].T + x @ p["A"].T + p["b"])


def readout(z, x, p):
    return z @ p["w"] + x @ p["a"] + p["c"]


@torch.no_grad()
def solve_equilibrium(x, p, q, state_atol=1e-12, max_iter=1000):
    """Stops when tau/(1-q) <= state_atol, not merely when tau is small.

    The bound holds in exact arithmetic; it does not include rounding error.
    The residual is evaluated at the state actually returned.
    """
    if not 0 <= q < 1:
        raise ValueError("0 <= q < 1 is required.")
    norm_W = float(p["W"].abs().sum(dim=1).max())
    if norm_W > q + 1e-14:
        raise ValueError("W violates the declared contraction bound.")
    z = torch.zeros((len(x), len(p["w"])), dtype=x.dtype, device=x.device)
    for iteration in range(max_iter + 1):
        next_z = state_map(z, x, p)
        residual = (next_z - z).abs().amax(dim=1)
        if float(residual.max()) / (1 - q) <= state_atol:
            return z, {
                "iterations": iteration,
                "max_residual": float(residual.max()),
                "state_error_bound": float(residual.max()) / (1 - q),
                "output_error_bound": float(p["w"].abs().sum()) * float(residual.max()) / (1 - q),
            }
        z = next_z
    raise RuntimeError("The equilibrium did not reach the tolerance. Check q, max_iter and float64.")


@torch.no_grad()
def implicit_input_derivatives(x, z, p):
    """Implicit-derivative formulas (Proposition A.3) evaluated at the approximate state z.

    D: (N,m), B: (N,m,m), J: (N,m,2), K: (N,m,2,2).
    Linear systems are solved; B^{-1} is never formed explicitly.
    Evaluation only: this block does not implement training gradients.
    """
    W, A, w = p["W"], p["A"], p["w"]
    s = z @ W.T + x @ A.T + p["b"]
    D = torch.sigmoid(s)
    second = D * (1 - D)
    m = W.shape[0]
    B = torch.eye(m, dtype=x.dtype, device=x.device)[None] - D[:, :, None] * W[None]
    rhs_J = D[:, :, None] * A[None]
    J = torch.linalg.solve(B, rhs_J)
    # V_i = W j_i + A_i.
    V = torch.einsum("ab,nbk->nak", W, J) + A[None]
    rhs_K = second[:, :, None, None] * V[:, :, :, None] * V[:, :, None, :]
    K = torch.linalg.solve(B, rhs_K.reshape(len(x), m, 4)).reshape(len(x), m, 2, 2)
    gradient = torch.einsum("m,nmi->ni", w, J) + p["a"]
    hessian = torch.einsum("m,nmij->nij", w, K)
    linear_J_residual = (B @ J - rhs_J).abs().max()
    linear_K_residual = (B @ K.reshape(len(x), m, 4) - rhs_K.reshape(len(x), m, 4)).abs().max()
    return readout(z, x, p), gradient, hessian, {
        "linear_J_residual": float(linear_J_residual),
        "linear_K_residual": float(linear_K_residual),
    }


def unrolled_output(x_single, p, steps):
    z = torch.zeros_like(p["w"])
    for _ in range(steps):
        z = softplus(p["W"] @ z + p["A"] @ x_single + p["b"])
    return p["w"] @ z + p["a"] @ x_single + p["c"]


def autograd_unrolled(x, p, steps):
    """Autograd through an explicit weight-tied network; does not use B, J or K."""
    values, gradients, hessians = [], [], []
    for point in x:
        point = point.detach().clone().requires_grad_(True)
        fun = lambda t: unrolled_output(t, p, steps)
        value = fun(point)
        grad = torch.autograd.grad(value, point)[0]
        hess = torch.autograd.functional.hessian(fun, point)
        values.append(value.detach())
        gradients.append(grad.detach())
        hessians.append(hess.detach())
    return torch.stack(values), torch.stack(gradients), torch.stack(hessians)


def compare(actual, reference, cfg):
    diff = (actual - reference).abs()
    tolerance = cfg.check_atol + cfg.check_rtol * reference.abs()
    return {
        "max_abs_error": float(diff.max()),
        "scaled_error": float((diff / tolerance).max()),
        "pass": bool(torch.all(diff <= tolerance)),
    }


def zero_W_check(cfg):
    """Exact case without recurrence: z* = softplus(Ax+b), explicit Hessian."""
    p = make_parameters(cfg.width, cfg.q, 101)
    p["W"] = torch.zeros_like(p["W"])
    x = sample_disk(7, 102)
    z, _ = solve_equilibrium(x, p, cfg.q, cfg.state_atol, cfg.max_iter)
    u, grad, hess, _ = implicit_input_derivatives(x, z, p)
    s = x @ p["A"].T + p["b"]
    d = torch.sigmoid(s)
    u_ref = softplus(s) @ p["w"] + x @ p["a"] + p["c"]
    g_ref = (d * p["w"]) @ p["A"] + p["a"]
    h_ref = torch.einsum("nm,mi,mj->nij", d * (1-d) * p["w"], p["A"], p["A"])
    return {"u": compare(u, u_ref, cfg), "gradient": compare(grad, g_ref, cfg),
            "hessian": compare(hess, h_ref, cfg)}


@torch.no_grad()
def convergence_rows(x, p, z_reference, ref_info, cfg, seed):
    """E1: errors against a numerical (not exact) reference equilibrium."""
    selected = {0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 128}
    z = torch.zeros_like(z_reference)
    C = float(state_map(z, x, p).abs().max())
    rows = []
    for L in range(max(selected) + 1):
        next_z = state_map(z, x, p)
        if L in selected:
            tau = float((next_z - z).abs().max())
            difference = float((z - z_reference).abs().max())
            # Cada cota es respecto al equilibrio exacto; la referencia tiene error propio.
            rows.append({
                "seed": seed, "L": L, "state_difference_to_reference": difference,
                "output_difference_to_reference": float((readout(z, x, p)-readout(z_reference, x, p)).abs().max()),
                "residual": tau, "posteriori_bound": tau / (1-cfg.q),
                "priori_bound": cfg.q**L * C / (1-cfg.q),
                "reference_state_bound": ref_info["state_error_bound"],
            })
        z = next_z
    return rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed in sorted(set(r["seed"] for r in rows)):
        group = [r for r in rows if r["seed"] == seed]
        depths = [r["L"] for r in group]
        positive = lambda key: [r[key] if r[key] > 0 else np.nan for r in group]
        axes[0].semilogy(depths, positive("state_difference_to_reference"), "o-", label=f"seed {seed}")
    first = [r for r in rows if r["seed"] == rows[0]["seed"]]
    for key, label, style in [
        ("state_difference_to_reference", "Difference to reference", "o-"),
        ("priori_bound", "A priori bound", "--"),
        ("posteriori_bound", "Residual bound", ":"),
    ]:
        axes[1].semilogy([r["L"] for r in first], [r[key] if r[key]>0 else np.nan for r in first], style, label=label)
    axes[0].set_title("E1: fixed parameters, three seeds")
    axes[1].set_title(f"Bounds and reference: seed {rows[0]['seed']}")
    for ax in axes:
        ax.set_xlabel("Iteraciones L")
        ax.set_ylabel("Infinity norm (sample maximum)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_experiments(output="results_e0_e1", cfg=None):
    cfg = Config() if cfg is None else cfg
    if cfg.width < 1 or not 0 <= cfg.q < 1 or not cfg.seeds:
        raise ValueError("Check width, q and seeds.")
    if not 1 <= cfg.n_derivative_points <= cfg.n_points:
        raise ValueError("n_derivative_points debe estar entre 1 y n_points.")
    if cfg.unroll_steps < 1 or cfg.state_atol <= 0 or cfg.reference_state_atol <= 0:
        raise ValueError("Steps and tolerances must be positive.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    summary = {
        "scope": "E0-E1; no training and no PDE solution results",
        "config": asdict(cfg),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                        "device": "cpu", "dtype": "float64", "platform": platform.platform()},
        "zero_W_check": zero_W_check(cfg), "runs": [],
    }
    try:
        summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except NameError:
        summary["script_sha256"] = "Notebook: embedded code; keep the executed notebook as well."
    all_rows, point_rows = [], []
    for seed in cfg.seeds:
        p = make_parameters(cfg.width, cfg.q, seed)
        x = sample_disk(cfg.n_points, seed + 1000)
        z, info = solve_equilibrium(x, p, cfg.q, cfg.state_atol, cfg.max_iter)
        z_ref, ref_info = solve_equilibrium(x, p, cfg.q, cfg.reference_state_atol, cfg.max_iter)
        u, grad, hess, linear_info = implicit_input_derivatives(x, z, p)
        n = cfg.n_derivative_points
        reference = autograd_unrolled(x[:n], p, cfg.unroll_steps)
        reference_deeper = autograd_unrolled(x[:n], p, 2 * cfg.unroll_steps)
        checks = {name: compare(value[:n], target, cfg) for name, value, target in
                  zip(("u", "gradient", "hessian"), (u, grad, hess), reference_deeper)}
        depth_checks = {name: compare(v, target, cfg) for name, v, target in
                        zip(("u", "gradient", "hessian"), reference, reference_deeper)}
        symmetric = 0.5 * (hess + hess.transpose(-1, -2))
        min_eig = float(torch.linalg.eigvalsh(symmetric).min())
        symmetry_error = float((hess-hess.transpose(-1,-2)).abs().max())
        norm_W = float(p["W"].abs().sum(dim=1).max())
        norm_A = float(torch.linalg.vector_norm(p["A"], dim=1).max())
        slope_bound = float(p["w"].abs().sum()) * norm_A / (1-cfg.q) + float(torch.linalg.vector_norm(p["a"]))
        max_grad = float(torch.linalg.vector_norm(grad, dim=1).max())
        rows = convergence_rows(x, p, z_ref, ref_info, cfg, seed)
        all_rows.extend(rows)
        roundoff_slack = 100 * torch.finfo(DTYPE).eps * max(1.0, float(z_ref.abs().max()))
        bounds_ok = all(r["state_difference_to_reference"] <=
                        min(r["priori_bound"], r["posteriori_bound"]) + r["reference_state_bound"] + roundoff_slack
                        for r in rows)
        gates = {
            "admissible_parameters": bool((p["W"] >= 0).all() and (p["w"] >= 0).all() and norm_W <= cfg.q + 1e-14),
            "derivatives_agree": all(v["pass"] for v in checks.values()),
            "unrolled_reference_stable": all(v["pass"] for v in depth_checks.values()),
            "hessian_symmetric": symmetry_error <= cfg.check_atol,
            "sampled_convexity": min_eig >= -cfg.convexity_atol,
            "sampled_slope_bound": max_grad <= slope_bound + cfg.check_atol,
            "state_bounds_numerically_consistent": bounds_ok,
            "linear_systems_solved": max(linear_info.values()) <= cfg.check_atol,
        }
        run = {"seed": seed, "norm_W_infinity": norm_W, "equilibrium": info,
               "reference_equilibrium": ref_info, "derivative_checks": checks,
               "reference_depth_checks": depth_checks, "linear_solve": linear_info,
               "hessian_symmetry_error": symmetry_error, "min_sampled_hessian_eigenvalue": min_eig,
               "input_lipschitz_bound": slope_bound, "max_sampled_gradient_norm": max_grad,
               "checks": gates, "pass": all(gates.values())}
        summary["runs"].append(run)
        for k in range(len(x)):
            point_rows.append({"seed": seed, "point": k, "x1": float(x[k,0]), "x2": float(x[k,1]),
                "u": float(u[k]), "grad1": float(grad[k,0]), "grad2": float(grad[k,1]),
                "h11": float(hess[k,0,0]), "h12": float(hess[k,0,1]),
                "h21": float(hess[k,1,0]), "h22": float(hess[k,1,1]),
                "det_hessian_untrained": float(torch.linalg.det(hess[k]))})
        np.savez(output / f"parameters_seed_{seed}.npz", **{k:v.numpy() for k,v in p.items()}, x=x.numpy())
        print(f"Seed {seed}: {'PASS' if run['pass'] else 'CHECK'} | "
              f"iterations={info['iterations']} | residual={info['max_residual']:.3e} | "
              f"Hessian error={checks['hessian']['max_abs_error']:.3e}")
    summary["all_checks_pass"] = all(r["pass"] for r in summary["runs"]) and all(
        c["pass"] for c in summary["zero_W_check"].values())
    summary["wall_time_seconds"] = time.perf_counter() - start
    summary["interpretation"] = (
        "PASS indicates local numerical checks, not a proof of training convergence. "
        "Maxima and minima are computed on the sample; they do not certify the whole disk. "
        "The unrolled reference is numerical; it is checked against twice as many iterations. "
        "The determinant belongs to an untrained network: it is not expected to equal 1. "
        "Residual bounds exclude rounding; linear systems are checked separately."
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output / "convergence.csv", all_rows)
    write_csv(output / "point_diagnostics.csv", point_rows)
    plot_convergence(all_rows, output / "convergence.png")
    human = ["E0-E1: VALIDATION, NO TRAINING", "",
             f"Overall result: {'PASS' if summary['all_checks_pass'] else 'CHECK'}",
             f"Computation time: {summary['wall_time_seconds']:.2f} s", "", summary["interpretation"], ""]
    for run in summary["runs"]:
        human.append(f"Seed {run['seed']}: {run['checks']}")
    (output / "summary.txt").write_text("\n".join(human), encoding="utf-8")
    archive = shutil.make_archive(str(output.resolve()), "zip", root_dir=output)
    print(f"\nOverall result: {'PASS' if summary['all_checks_pass'] else 'CHECK'}")
    print(f"Output archive: {archive}")
    return summary, Path(archive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results_e0_e1")
    args = parser.parse_args()
    summary, _ = run_experiments(args.out, Config())
    if not summary["all_checks_pass"]:
        raise SystemExit("E0-E1 validation: some checks failed; see summary.json")
