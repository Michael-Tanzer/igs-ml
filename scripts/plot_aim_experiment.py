"""Plot all metrics from an AIM experiment across all runs.

Usage:
    python scripts/plot_aim_experiment.py [--experiment malaria_patch_baseline] [--aim-repo /data/mtanzer/aim/] [--out-dir tmp/]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as _cm

import aim


# ── helpers ──────────────────────────────────────────────────────────────────

def get_metric_data(run, metric_name: str, context):
    """Return (steps, values) arrays for a metric in a run, or (None, None)."""
    try:
        metric = run.get_metric(metric_name, context)
        if metric is None:
            return None, None
        steps, (values, epochs, timestamps) = metric.data.numpy()
        if len(steps) == 0:
            return None, None
        return steps, values
    except Exception:
        return None, None


def full_metric_name(name: str, ctx) -> str:
    """Reconstruct the full metric name from name + context subset."""
    subset = ctx.to_dict().get("subset", None)
    if subset:
        return f"{subset}/{name}"
    return name


def plot_metric_grid(runs_data, metric_names, title, out_path, ncols=3):
    """
    runs_data: list of dict with keys 'name', 'color', and full_metric_name -> (steps, values)
    """
    nrows = (len(metric_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, metric in enumerate(metric_names):
        ax = axes[idx // ncols][idx % ncols]
        has_data = False
        for rd in runs_data:
            steps, vals = rd.get(metric, (None, None))
            if steps is not None and len(steps) >= 2:
                ax.plot(steps, vals, color=rd["color"], label=rd["name"], linewidth=1.2)
                has_data = True
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if not has_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

    # Hide unused axes
    for idx in range(len(metric_names), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Single legend below the figure
    handles = [plt.Line2D([0], [0], color=rd["color"], linewidth=2) for rd in runs_data]
    labels = [rd["name"] for rd in runs_data]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(runs_data), 4),
               bbox_to_anchor=(0.5, 0), fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="malaria_patch_baseline")
    parser.add_argument("--aim-repo", default="/data/mtanzer/aim/")
    parser.add_argument("--out-dir", default="tmp/")
    parser.add_argument("--min-steps", type=int, default=5,
                        help="Skip runs with fewer than this many steps (likely failed)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cmap = matplotlib.colormaps["tab10"]
    except AttributeError:
        cmap = _cm.get_cmap("tab10")

    repo = aim.Repo(args.aim_repo)
    query = f"run.experiment == '{args.experiment}'"
    print(f"Querying: {query}")
    all_run_collections = list(repo.query_runs(query).iter_runs())
    print(f"Found {len(all_run_collections)} run(s)")

    # Build runs_data list
    all_metric_keys: set = set()
    runs_data = []

    for i, rc in enumerate(all_run_collections):
        run = rc.run
        model_name = run.name or run.hash[:8]
        color = cmap(i / max(len(all_run_collections) - 1, 1))

        # Collect all (name, ctx) metric info pairs, skip system metrics
        infos = [(n, ctx) for n, ctx, _ in run.iter_metrics_info()
                 if not n.startswith("__system")]

        # Check if run has enough data using train/loss_step
        train_loss_infos = [(n, ctx) for n, ctx in infos
                            if n == "loss_step" and ctx.to_dict().get("subset") == "train"]
        if train_loss_infos:
            steps, _ = get_metric_data(run, "loss_step", train_loss_infos[0][1])
            n_steps = 0 if steps is None else len(steps)
        else:
            n_steps = 0

        if n_steps < args.min_steps:
            print(f"  SKIP {model_name} (hash={run.hash[:8]}): "
                  f"train/loss has {n_steps} steps (< {args.min_steps})")
            continue

        print(f"  Run: {model_name} | train/loss steps: {n_steps} | "
              f"metric types: {len(infos)}")

        rd: dict = {"name": model_name, "color": color}
        for mname, ctx in infos:
            full_name = full_metric_name(mname, ctx)
            s, v = get_metric_data(run, mname, ctx)
            if s is not None:
                rd[full_name] = (s, v)
                all_metric_keys.add(full_name)
        runs_data.append(rd)

    if not runs_data:
        print("No valid runs found. Exiting.")
        return

    print(f"\nPlotting {len(runs_data)} runs, {len(all_metric_keys)} unique metrics...")

    # ── Figure 1: Loss curves ─────────────────────────────────────────────────
    loss_metrics = sorted(m for m in all_metric_keys if "loss" in m.lower())
    if loss_metrics:
        plot_metric_grid(runs_data, loss_metrics, "Loss Curves",
                         out_dir / "loss_curves.png", ncols=3)

    # ── Figure 2: Patch-level classification metrics ─────────────────────────
    patch_keywords = ["accuracy", "auroc", "f1", "precision", "recall",
                      "specificity", "avg_precision", "average_precision"]
    patch_metrics = sorted(
        m for m in all_metric_keys
        if any(kw in m.lower() for kw in patch_keywords)
        and "opt_" not in m and "patient" not in m
    )
    if patch_metrics:
        plot_metric_grid(runs_data, patch_metrics, "Patch-level Classification Metrics",
                         out_dir / "patch_metrics.png", ncols=4)

    # ── Figure 3: Optimal threshold metrics ──────────────────────────────────
    opt_metrics = sorted(m for m in all_metric_keys if "opt_" in m)
    if opt_metrics:
        plot_metric_grid(runs_data, opt_metrics, "Optimal Threshold Metrics",
                         out_dir / "opt_threshold_metrics.png", ncols=2)

    # ── Figure 4: Patient-level metrics ──────────────────────────────────────
    patient_metrics = sorted(m for m in all_metric_keys if "patient" in m)
    if patient_metrics:
        plot_metric_grid(runs_data, patient_metrics, "Patient-level Metrics",
                         out_dir / "patient_metrics.png", ncols=3)

    # ── Figure 5: Everything else (lr, etc.) ─────────────────────────────────
    used = set(loss_metrics) | set(patch_metrics) | set(opt_metrics) | set(patient_metrics)
    other_metrics = sorted(m for m in all_metric_keys if m not in used)
    if other_metrics:
        plot_metric_grid(runs_data, other_metrics, "Other Metrics",
                         out_dir / "other_metrics.png", ncols=4)

    print("\nDone. Figures saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
