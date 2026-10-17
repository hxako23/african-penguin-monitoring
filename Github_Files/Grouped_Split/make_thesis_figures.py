#!/usr/bin/env python3
# Figures for the thesis "Lightweight Object Detection for African Penguin Population Monitoring".


#    python3 make_thesis_figures.py              # trade-off plot and training curves, from saved results
#    python3 make_thesis_figures.py --density    # also counts penguins by crowding (Dane's density split, GPU, ~5 min)
#    python3 make_thesis_figures.py --examples   # also draws YOLOv11n detections on four African Penguin test photos
#    python3 make_thesis_figures.py --all        # everything

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- paths on the Dell PC
HOME = os.path.expanduser("~")
RESULTS = os.environ.get("THESIS_RESULTS", f"{HOME}/hluma_results")
DATASET = os.environ.get("THESIS_DATASET", f"{HOME}/penguin_dataset_702010")
OUT = f"{RESULTS}/thesis_figures"

GROUPED_RUNS = {   # Experiment 1.2
    "YOLOv11n": f"{RESULTS}/runs_702010/yolov11n",
    "YOLOv11m": f"{RESULTS}/runs_702010/yolov11m",
    "YOLOv26n": f"{RESULTS}/runs_702010/yolov26n",
    "YOLOv26m": f"{RESULTS}/runs_702010/yolov26m",
}
ORIGINAL_RUNS = {  # Experiment 1.1
    "YOLOv11n": f"{RESULTS}/old_split/runs/yolov11n-3",
    "YOLOv11m": f"{RESULTS}/old_split/runs/yolov11m",
    "YOLOv26n": f"{RESULTS}/old_split/runs/yolov26n",
    "YOLOv26m": f"{RESULTS}/old_split/runs/yolo26m-2",
}
ORDER = ["YOLOv11n", "YOLOv11m", "YOLOv26n", "YOLOv26m"]

# ---------------------------------------------------------------- one style for every figure
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
    "lines.linewidth": 1.4, "savefig.dpi": 300,
})
# Colour choice. Change PALETTE here, or run with --palette NAME.
PALETTES = {
    "blue_green":   {"11": "#0072B2", "26": "#009E73"},   # Okabe-Ito blue and green
    "navy_crimson": {"11": "#1F4E79", "26": "#B2182B"},   # one colour per YOLO generation
    "black_grey":   {"11": "#111111", "26": "#8C8C8C"},   # prints the same in colour or black and white
    "teal_purple":  {"11": "#00798C", "26": "#6A3D9A"},
    "four":         {"YOLOv11n": "#1F4E79", "YOLOv11m": "#B2182B", "YOLOv26n": "#6A3D9A", "YOLOv26m": "#4D4D4D"},
}
PALETTE = "blue_green"


def colour_of(model):
    pal = PALETTES[PALETTE]
    return pal.get(model) or pal["11" if "11" in model else "26"]


def style(model):
    """YOLOv11 blue with circles, YOLOv26 green with triangles; nano dashed, medium solid."""
    colour = colour_of(model)
    marker = "o" if "11" in model else "^"
    ls = "--" if model.endswith("n") else "-"
    return colour, marker, ls


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT}/{name}.pdf")


def read_csv(path):
    with open(path) as f:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


# ---------------------------------------------------------------- 1. accuracy against CPU latency
def tradeoff():
    det_csv, lat_csv = f"{RESULTS}/detection_702010.csv", f"{RESULTS}/latency_702010.csv"
    if not (os.path.exists(det_csv) and os.path.exists(lat_csv)):
        print("  skipped: detection_702010.csv or latency_702010.csv not found")
        return
    acc = {r["model"]: float(r["map50_95"]) for r in read_csv(det_csv) if r["test_set"] == "all"}
    lat = {(r["model"], r["runtime"]): float(r["mean_ms"]) for r in read_csv(lat_csv)}

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for gen in ("11", "26"):
        n, m = f"YOLOv{gen}n", f"YOLOv{gen}m"
        for runtime, ls in (("OpenVINO", "-"), ("PyTorch", "--")):
            xs, ys = [lat[(n, runtime)], lat[(m, runtime)]], [acc[n], acc[m]]
            ax.plot(xs, ys, ls=ls, color=colour_of(n) if PALETTE != "four" else "0.6", lw=1.2, zorder=1)
            for x, y, size_marker, mod in zip(xs, ys, ("o", "s"), (n, m)):
                c = colour_of(mod)
                face = c if runtime == "OpenVINO" else "white"
                ax.plot(x, y, marker=size_marker, ms=7, mfc=face, mec=c, mew=1.5, ls="none", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([15, 20, 30, 50, 100, 150])
    ax.set_xticklabels(["15", "20", "30", "50", "100", "150"])
    ax.minorticks_off()
    ax.set_xlabel("Mean CPU latency per image (ms, log scale)")
    ax.set_ylabel("mAP@50:95 (full test set)")
    if PALETTE == "four":
        handles = [Line2D([], [], color=colour_of(k), marker="o" if k.endswith("n") else "s", ls="none", ms=7, label=k)
                   for k in ORDER]
    else:
        handles = [
            Line2D([], [], color=colour_of("YOLOv11n"), lw=1.2, label="YOLOv11"),
            Line2D([], [], color=colour_of("YOLOv26n"), lw=1.2, label="YOLOv26"),
            Line2D([], [], color="0.3", marker="o", ls="none", ms=7, label="nano"),
            Line2D([], [], color="0.3", marker="s", ls="none", ms=7, label="medium")]
    handles += [
        Line2D([], [], color="0.3", ls="-", marker="o", ms=6, mfc="0.3", label="OpenVINO"),
        Line2D([], [], color="0.3", ls="--", marker="o", ms=6, mfc="white", label="PyTorch")]
    ax.legend(handles=handles, ncol=3, loc="lower right", frameon=True, framealpha=1, edgecolor="0.8")
    fig.tight_layout()
    save(fig, "tradeoff_map_latency")
    for k in ORDER:
        print(f"    {k}: mAP@50:95 {acc[k]:.4f}, OpenVINO {lat[(k, 'OpenVINO')]:.2f} ms, PyTorch {lat[(k, 'PyTorch')]:.2f} ms")


# ---------------------------------------------------------------- 2. training curves for each split
def curves(runs, name, split_label):
    found = {m: p for m, p in runs.items() if os.path.exists(f"{p}/results.csv")}
    for m in runs:
        if m not in found:
            print(f"  {split_label}: no results.csv for {m} ({runs[m]}), skipped")
    if not found:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
    for m in ORDER:
        if m not in found:
            continue
        rows = read_csv(f"{found[m]}/results.csv")
        ep = [int(float(r["epoch"])) for r in rows]
        loss = [float(r["train/box_loss"]) for r in rows]
        mp = [float(r["metrics/mAP50-95(B)"]) for r in rows]
        colour, marker, ls = style(m)
        kw = dict(color=colour, ls=ls, marker=marker, markevery=10, ms=4, label=m)
        a.plot(ep, loss, **kw)
        b.plot(ep, mp, **kw)
        best = max(range(len(mp)), key=lambda i: mp[i])
        print(f"    {split_label} {m}: {len(ep)} epochs, best val mAP@50:95 {mp[best]:.4f} at epoch {ep[best]}, "
              f"train box loss {loss[0]:.3f} -> {loss[-1]:.3f}")
    a.set(xlabel="Epoch", ylabel="Box loss", title="(a) Training box loss")
    b.set(xlabel="Epoch", ylabel="mAP@50:95", title="(b) Validation mAP@50:95")
    h, l = b.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, name)


# ---------------------------------------------------------------- 3. density split (Dane's suggestion)
BINS = [("1 to 2", 1, 2), ("3 to 10", 3, 10), ("over 10", 11, 10**6)]


def true_count(img):
    lbl = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    with open(lbl) as f:
        return sum(1 for line in f if line.strip())


def density():
    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    imgs = [p for p in sorted(glob.glob(f"{DATASET}/test/images/*")) if true_count(p) > 0]
    truth = {p: true_count(p) for p in imgs}
    rows = []
    for m in ORDER:
        weights = f"{GROUPED_RUNS[m]}/weights/best.pt"
        if not os.path.exists(weights):
            print(f"  {m}: {weights} not found, skipped")
            continue
        model = YOLO(weights)
        pred = {p: len(model(p, conf=0.25, device=device, verbose=False)[0].boxes) for p in imgs}
        for name, lo, hi in BINS:
            group = [p for p in imgs if lo <= truth[p] <= hi]
            t = [truth[p] for p in group]
            q = [pred[p] for p in group]
            row = {"model": m, "group": name, "photos": len(group), "penguins": sum(t),
                   "mae": round(sum(abs(a - b) for a, b in zip(q, t)) / len(t), 2),
                   "within_5pct": round(100 * sum(abs(a - b) / b <= 0.05 for a, b in zip(q, t)) / len(t), 1),
                   "total_error_pct": round(100 * (sum(q) - sum(t)) / sum(t), 2)}
            print("   ", row)
            rows.append(row)
    if not rows:
        return
    with open(f"{RESULTS}/density_702010.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {RESULTS}/density_702010.csv")
    density_outputs(rows)


def density_outputs(rows=None):
    path = f"{RESULTS}/density_702010.csv"
    if rows is None:
        if not os.path.exists(path):
            print("  skipped: density_702010.csv not found (run with --density)")
            return
        rows = read_csv(path)
    groups = [g for g, _, _ in BINS]
    models = [m for m in ORDER if any(r["model"] == m for r in rows)]
    val = {(r["model"], r["group"]): r for r in rows}
    if not all((models[0], g) in val for g in groups):
        print("  skipped: density_702010.csv uses old groups, rerun with --density")
        return

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    width = 0.2
    for i, m in enumerate(models):
        colour, _, _ = style(m)
        hatch = "///" if m.endswith("n") else None
        xs = [j + (i - 1.5) * width for j in range(len(groups))]
        a.bar(xs, [float(val[(m, g)]["mae"]) for g in groups], width, color=colour if not hatch else "white",
              edgecolor=colour, hatch=hatch, lw=1.2, label=m)
        b.bar(xs, [float(val[(m, g)]["total_error_pct"]) for g in groups], width, color=colour if not hatch else "white",
              edgecolor=colour, hatch=hatch, lw=1.2, label=m)
    labels = [f"{g}\n({val[(models[0], g)]['photos']} photos)" for g in groups]
    for ax in (a, b):
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Penguins per photograph")
        ax.grid(axis="x", visible=False)
    a.set(ylabel="MAE (penguins per photograph)", title="(a) Error per photograph")
    b.axhspan(-5, 5, color="0.88", zorder=0)
    b.text(0.99, 0.97, "shaded: within 5%", transform=b.transAxes, ha="right", va="top", fontsize=9, color="0.3")
    b.axhline(0, color="0.2", lw=0.8)
    b.set(ylabel="Total count error (%)", title="(b) Error in the total count")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "density_split")

    with open(f"{OUT}/density_table.tex", "w") as f:
        f.write("\\begin{tabular}{llccc}\n\\toprule\n"
                "\\textbf{Penguins per photo} & \\textbf{Model} & \\textbf{MAE} & \\textbf{Within 5\\%} & \\textbf{Total error} \\\\\n\\midrule\n")
        for gi, g in enumerate(groups):
            for mi, m in enumerate(models):
                r = val[(m, g)]
                first = f"{g} ({r['photos']} photos)" if mi == 0 else ""
                f.write(f"{first} & {m} & {float(r['mae']):.2f} & {float(r['within_5pct']):.1f}\\% & "
                        f"${float(r['total_error_pct']):+.1f}\\%$ \\\\\n")
            if gi < len(groups) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  saved {OUT}/density_table.tex")


# ---------------------------------------------------------------- 4. example detections
def examples(targets=(2, 7, 15, 40)):
    import torch
    from matplotlib.patches import Rectangle
    from PIL import Image
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    weights = f"{GROUPED_RUNS['YOLOv11n']}/weights/best.pt"
    if not os.path.exists(weights):
        print(f"  skipped: {weights} not found")
        return
    model = YOLO(weights)
    african = [p for p in sorted(glob.glob(f"{DATASET}/test/images/*"))
               if os.path.basename(p).startswith("african-penguin")]
    counts = {p: true_count(p) for p in african}
    chosen = []
    for t in targets:   # the photo whose true count is closest to each target
        p = min((p for p in african if p not in chosen), key=lambda p: abs(counts[p] - t))
        chosen.append(p)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    for ax, p in zip(axes.flat, chosen):
        res = model(p, conf=0.25, device=device, verbose=False)[0]
        ax.imshow(Image.open(p).convert("RGB"))
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#00E000", lw=1.3))
        ax.set_title(f"True count {counts[p]}, predicted {len(res.boxes)}")
        ax.axis("off")
        print(f"    {os.path.basename(p)}: true {counts[p]}, predicted {len(res.boxes)}")
    fig.tight_layout()
    save(fig, "example_detections")


# ---------------------------------------------------------------- 5. tuning trials (Experiment 4.2)
def spearman(x, y):
    import numpy as np
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def tuning():
    import json
    import numpy as np
    folder = f"{RESULTS}/ray_results_tune702010_yolo11n"
    rows = []
    for p in glob.glob(f"{folder}/**/params.json", recursive=True):
        params = json.load(open(p))
        r = os.path.join(os.path.dirname(p), "result.json")
        last = None
        if os.path.exists(r):
            for line in open(r):
                if line.strip():
                    last = json.loads(line)
        key = next((k for k in (last or {}) if "mAP50-95" in k or "map50_95" in k.lower()), None)
        if key:
            rows.append((params["lr0"], params["momentum"], params["weight_decay"], float(last[key])))
    if not rows:
        print(f"  skipped: no trial results found in {folder}")
        return
    lr, mom, wd, score = (np.array(c) for c in zip(*sorted(rows)))
    fit = np.polyfit(np.log10(lr), score, 1)
    rest = (score - np.polyval(fit, np.log10(lr))) * 100        # in points
    r2 = 1 - np.var(score - np.polyval(fit, np.log10(lr))) / np.var(score)
    colour = colour_of("YOLOv11n")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    a = axes[0]
    a.scatter(lr, score, s=28, color=colour, zorder=3)
    xs = np.logspace(np.log10(lr.min()), np.log10(lr.max()), 50)
    a.plot(xs, np.polyval(fit, np.log10(xs)), color="0.4", lw=1, ls="--")
    a.set_xscale("log")
    a.set(xlabel="Learning rate (log scale)", ylabel="Validation mAP@50:95",
          title=f"(a) Learning rate, $\\rho$ = {spearman(lr, score):+.2f}")
    for ax, x, name, tag, logx in ((axes[1], mom, "Momentum", "(b)", False), (axes[2], wd, "Weight decay", "(c)", True)):
        ax.scatter(x, rest, s=28, color=colour, zorder=3)
        ax.axhline(0, color="0.4", lw=1, ls="--")
        if logx:
            ax.set_xscale("log")
        ax.set(xlabel=name + (" (log scale)" if logx else ""), title=f"{tag} {name}, $\\rho$ = {spearman(x, rest):+.2f}")
    axes[1].set_ylabel("Score left after (a), in points")
    fig.tight_layout()
    save(fig, "tuning_trials")
    print(f"    {len(rows)} trials; learning rate explains R^2 = {r2:.2f} of the variation; "
          f"rho(lr, score) = {spearman(lr, score):+.2f}, rho(momentum, rest) = {spearman(mom, rest):+.2f}, "
          f"rho(weight decay, rest) = {spearman(wd, rest):+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", action="store_true", help="count penguins by crowding (uses the GPU)")
    ap.add_argument("--examples", action="store_true", help="draw YOLOv11n detections on four test photos")
    ap.add_argument("--all", action="store_true", help="everything")
    ap.add_argument("--palette", choices=list(PALETTES), help="colour scheme (default: %(default)s)")
    args = ap.parse_args()
    if args.palette:
        PALETTE = args.palette
    print("Trade-off plot")
    tradeoff()
    print("Training curves")
    curves(GROUPED_RUNS, "training_curves_grouped", "grouped")
    curves(ORIGINAL_RUNS, "training_curves_original", "original")
    print("Tuning trials")
    tuning()
    print("Density split")
    density() if (args.density or args.all) else density_outputs()
    if args.examples or args.all:
        print("Example detections")
        examples()
    print(f"\nDone. Upload the PDFs from {OUT} to Chapters/Images/ in Overleaf.")
