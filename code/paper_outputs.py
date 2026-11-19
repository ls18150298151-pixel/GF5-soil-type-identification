"""Build review tables and English analysis figures from locally regenerated results."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from settings import CLASS_NAMES, CONFIG, INPUT, MODE, PROTOCOL, ROOT, resolve_path
from formal_runner import FAMILIES, checkpoint_path
from v2_features import readj, verify_identity

BRANCHES = ("S0", "N59", "I63", "FALL", "FS")
BLOCKS = ("OR", "CR", "FDR")
REDUCERS = ("CARS", "PCA", "PLS-VIP", "SPA")
FAMILY_LABEL = {"LightGBM": "LightGBM", "XGBoost": "XGBoost", "RandomForest": "RF", "SVM": "SVM"}
COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#b68100", "#555555")
METRIC_LABEL = {"overall_accuracy": "OA", "macro_f1": "Macro-F1", "balanced_accuracy": "BA"}


def output_table(rows, directory, name):
    pd.DataFrame(rows).to_csv(directory / f"{name}.csv", index=False, encoding="utf-8")


def selected_tables(directory, pool, selected):
    # Older formal result logs refer to the composite spectral candidate ID for S0.
    # Resolve it via the winning spectral record to the underlying model configuration.
    registry = readj(PROTOCOL / "spectral_candidate_registry_972.json")["candidates"]
    spectral_to_model = {row["spectral_candidate_id"]: row["model_candidate_id"] for row in registry}
    lookup = {row["candidate_id"]: row for row in pool}
    rows, backbone_rows, used_configs, fs_sets = [], [], set(), {}
    for branch in BRANCHES:
        subset = sorted((row for row in selected if row["policy"] == "R" and row["branch"] == branch),
                        key=lambda row: row["outer_fold"])
        for row in subset:
            log = readj(ROOT / "fits" / f'{row["outer_fit_key"]}.json')
            info = log["feature_info"]
            candidate = spectral_to_model.get(row["candidate_id"], row["candidate_id"])
            used_configs.add(candidate)
            outer = row["outer_fold"]
            if branch == "S0":
                method = info["reducer"]
                params = info["params"]
                backbone_rows.append({
                    "outer_fold": outer, "spectral_compression": method,
                    "model_configuration": candidate, "classifier": row["family"],
                    "outer_macro_f1": row["outer_macro_f1"],
                })
            elif branch in ("FALL", "FS"):
                path = checkpoint_path(["fusion", "R", outer, branch, row["family"]])
                backbone = readj(path)["backbone"]
                method, params = backbone["reducer"], backbone["reducer_params"]
            else:
                method, params = "", {}
            factors = info.get("actual_env_factors", [])
            env_columns = sum(len(v) for v in info.get("layout", {}).values())
            rows.append({
                "branch": branch, "outer_fold": outer, "spectral_compression": method,
                "spectral_parameters": json.dumps(params, sort_keys=True),
                "model_configuration": candidate, "weighting": lookup[candidate]["strategy"],
                "environmental_modules": "+".join(row["combo"] or []),
                "actual_environmental_factors": len(factors),
                "spectral_columns": info["output_dim"] - env_columns, "environmental_columns": env_columns,
                "total_input_dimension": info["output_dim"],
            })
            if branch == "FS":
                fs_sets[outer] = set(factors)
    if len(rows) != 25 or len(fs_sets) != 5:
        raise ValueError("Expected five selected pipelines for each main branch")
    output_table(rows, directory, "Table_S2")
    output_table(backbone_rows, directory, "Fig_07_backbone_by_fold")
    output_table([
        {"model_configuration": cid, "classifier": lookup[cid]["classifier"],
         "hyperparameters": json.dumps(lookup[cid]["params"], sort_keys=True)}
        for cid in sorted(used_configs)
    ], directory, "Table_S3")
    catalog = pd.read_csv(PROTOCOL / "factor_catalog_63.csv")
    fs_rows = []
    for i, row in enumerate(catalog.itertuples(index=False), 1):
        flags = {f"fold_{fold}": int(row.factor in fs_sets[fold]) for fold in range(1, 6)}
        if any(flags.values()):
            fs_rows.append(dict(catalog_no=i, factor=row.factor, module=row.module,
                                **flags, retention_frequency=sum(flags.values())))
    output_table(fs_rows, directory, "Table_S4")
    output_table(pd.read_csv(PROTOCOL / "S1_factor_definitions.csv"), directory, "Table_S1")
    pairs = pd.DataFrame(rows).query("branch in ['FALL', 'FS']")
    output_table(pairs.drop(columns=["spectral_parameters"]), directory, "Table_5")
    return backbone_rows


def parameter_tables(directory, pool):
    for number, family in enumerate(("LightGBM", "XGBoost", "RandomForest", "SVM"), 1):
        sets = {}
        for candidate in pool:
            if candidate["classifier"] == family:
                params = candidate["params"]
                key = json.dumps(params, sort_keys=True)
                if key not in sets:
                    sets[key] = {"set": len(sets) + 1, **params}
        output_table(list(sets.values()), directory, f"Table_{number}")


def save_plot(fig, directory, number):
    fig.savefig(directory / f"Fig_{number:02}.png", dpi=200, bbox_inches="tight")
    fig.savefig(directory / f"Fig_{number:02}.svg", bbox_inches="tight")
    plt.close(fig)


def outline(ax, path, target):
    from pyproj import Transformer

    project = Transformer.from_crs("EPSG:4326", target, always_xy=True)
    body = readj(path)
    shapes = [feature["geometry"] for feature in body["features"]] if body["type"] == "FeatureCollection" else [
        body["geometry"] if body["type"] == "Feature" else body
    ]
    for item in shapes:
        if item["type"] == "Polygon":
            polygons = [item["coordinates"]]
        elif item["type"] == "MultiPolygon":
            polygons = item["coordinates"]
        else:
            raise ValueError("The optional study boundary must have Polygon or MultiPolygon geometry.")
        for polygon in polygons:
            for ring in polygon:
                array = np.asarray(ring, dtype=float)
                x, y = project.transform(array[:, 0], array[:, 1])
                ax.plot(x, y, color="#656565", linewidth=.8)


def plot_maps(directory, groups):
    config = CONFIG["inputs"]
    path = config.get("study_boundary")
    projected_crs = readj(INPUT / "preparation.json")["projected_crs"]
    if path:
        fig, ax = plt.subplots(figsize=(6, 5))
        outline(ax, resolve_path(path), projected_crs)
        ax.set(xlabel="Easting (m)", ylabel="Northing (m)", aspect="equal", title="Study-area outline")
        save_plot(fig, directory, 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    if path:
        outline(ax, resolve_path(path), projected_crs)
    reference = groups.query("is_real == 1")
    for code, soil in enumerate(CLASS_NAMES):
        pts = reference[reference.class_code_0based == code]
        ax.scatter(pts.easting, pts.northing, s=11, label=soil, color=COLORS[code])
    virtual = groups.query("is_virtual == 1")
    if not virtual.empty:
        ax.scatter(virtual.easting, virtual.northing, s=12, color="#555555", marker="^",
                   label="Virtual samples")
    ax.legend(fontsize=8, ncol=2)
    ax.set(xlabel="Easting (m)", ylabel="Northing (m)", aspect="equal", title="Soil sampling sites")
    save_plot(fig, directory, 2)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    for fold, ax in enumerate(axes.flat, 1):
        if fold > 5:
            ax.set_axis_off()
            break
        if path:
            outline(ax, resolve_path(path), projected_crs)
        is_val = reference.outer_fold == fold
        ax.scatter(reference.loc[~is_val, "easting"], reference.loc[~is_val, "northing"], color="#c0c0c0", s=7)
        val = reference[is_val]
        ax.scatter(val.easting, val.northing, c=[COLORS[c] for c in val.class_code_0based], s=15)
        ax.set(title=f"Outer fold {fold}", aspect="equal")
    fig.supxlabel("Easting (m)")
    fig.supylabel("Northing (m)")
    save_plot(fig, directory, 5)


def plot_spectra(directory, metadata):
    with np.load(INPUT / "modeling_inputs.npz", allow_pickle=False) as z:
        wave, segments = z["wavelength_nm"], z["segments"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
        for block, ax in zip(BLOCKS, axes):
            data = z[block]
            for code, soil in enumerate(CLASS_NAMES):
                values = data[(metadata.is_real == 1) & (metadata.class_code_0based == code)]
                mean = values.mean(axis=0)
                sd = values.std(axis=0, ddof=1)
                for segment in np.unique(segments):
                    m = segments == segment
                    ax.plot(wave[m], mean[m], color=COLORS[code], lw=1.2,
                            label=soil if segment == segments[0] else None)
                    ax.fill_between(wave[m], mean[m]-sd[m], mean[m]+sd[m], color=COLORS[code], alpha=.13)
            ax.set(title=block, xlabel="Wavelength (nm)")
        axes[0].legend(fontsize=7)
        save_plot(fig, directory, 6)


def plot_workflow(directory):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    stages = [
        "Reference samples, GF-5 imagery and environmental factor grids",
        "OR, CR and FDR spectra with candidate environmental factors",
        "2 km spatial blocks: five outer folds and four inner folds",
        "48 fixed spectral families and five input branches\nTraining-fold processing, selection and tuning",
        "Outer holdout predictions, paired contrasts and sensitivity analyses",
    ]
    y = np.linspace(.87, .13, len(stages))
    for i, (label, pos) in enumerate(zip(stages, y)):
        ax.text(.5, pos, label, va="center", ha="center",
                bbox=dict(boxstyle="round,pad=.75", facecolor="white", edgecolor="#333333"))
        if i:
            ax.annotate("", (.5, pos+.05), (.5, y[i-1]-.05),
                        arrowprops=dict(arrowstyle="->", color="#333333"))
    save_plot(fig, directory, 3)


def plot_performance(directory, metrics, paired, fixed, backbone):
    fig = plt.figure(figsize=(10, 6), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    maximum = max(metrics[f"{b}_{r}_{f}"]["macro_f1"] for b in BLOCKS for r in REDUCERS for f in FAMILIES)
    for block, ax in zip(BLOCKS, axes):
        values = np.array([[100*metrics[f"{block}_{r}_{f}"]["macro_f1"] for f in FAMILIES] for r in REDUCERS])
        heat = ax.imshow(values, vmin=0, vmax=max(1, 100*maximum), cmap="Blues")
        ax.set(xticks=range(4), xticklabels=[FAMILY_LABEL[f] for f in FAMILIES],
               yticks=range(4), yticklabels=REDUCERS, title=block)
        for y, x in np.ndindex(values.shape):
            ax.text(x, y, f"{values[y, x]:.1f}", ha="center", va="center",
                    color="white" if values[y, x] > 60*maximum else "black", fontsize=8)
    fig.colorbar(heat, ax=axes, label="Pooled Macro-F1 (%)", shrink=.8)
    ax = fig.add_subplot(grid[1, :])
    xs = [r["outer_fold"] for r in backbone]
    ys = [100*r["outer_macro_f1"] for r in backbone]
    ax.plot(xs, ys, marker="o", color=COLORS[0])
    for r, x_pos, y_pos in zip(backbone, xs, ys):
        name = FAMILY_LABEL[r["classifier"]]
        ax.annotate(f'{r["spectral_compression"]}–{name}', (x_pos, y_pos), (0, 7),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.margins(x=.1, y=.4)
    ax.set(xticks=range(1, 6), xlabel="Outer fold", ylabel="Macro-F1 (%)",
           title="FDR backbone selected by inner validation")
    save_plot(fig, directory, 7)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(5)
    for j, (metric, label) in enumerate(METRIC_LABEL.items()):
        ax.bar(x+(j-1)*.23, [100*metrics[b][metric] for b in BRANCHES], width=.23, label=label)
    ax.set(xticks=x, xticklabels=BRANCHES, ylabel="Pooled score (%)", ylim=(0, 100))
    ax.legend()
    save_plot(fig, directory, 8)

    order = ["environment_increment", "GF5_indices", "FDR_backbone", "all_GF5_increment", "FS_minus_FALL"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True, layout="constrained")
    for ax, (metric, label) in zip(axes, METRIC_LABEL.items()):
        rows = paired[paired.metric == metric].set_index("comparison").loc[order]
        x = np.arange(len(rows))
        ax.hlines(x, rows.conditional_block_bootstrap_low_pp, rows.conditional_block_bootstrap_high_pp,
                  color="#0072b2")
        ax.scatter(rows.difference_pp, x, color="#0072b2", zorder=2)
        ax.axvline(0, color="grey", linewidth=1)
        ax.set(title=label, xlabel="Difference (percentage points)")
    labels = [f"{a.split(':')[-1]} − {b.split(':')[-1]}" for a, b in zip(rows.A, rows.B)]
    axes[0].set(yticks=x, yticklabels=labels)
    axes[0].invert_yaxis()
    save_plot(fig, directory, 9)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, layout="constrained")
    for metric, ax in zip(("overall_accuracy", "macro_f1"), axes):
        for f in FAMILIES:
            sub = [row for row in fixed if row["classifier"] == f]
            ax.plot([r["outer_fold"] for r in sub], [100*r[metric] for r in sub],
                    marker="o", label=FAMILY_LABEL[f])
        ax.set(xticks=range(1, 6), title=METRIC_LABEL[metric], xlabel="Outer fold")
    axes[0].set_ylabel("FALL score (%)")
    axes[-1].legend(fontsize=8)
    save_plot(fig, directory, 10)

    cm = np.array(metrics["FALL"]["confusion_matrix"])
    fractions = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float),
                          where=cm.sum(axis=1, keepdims=True) > 0)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    heat = ax.imshow(fractions*100, vmin=0, vmax=100, cmap="Blues")
    ax.set(xticks=range(6), xticklabels=CLASS_NAMES, yticks=range(6), yticklabels=CLASS_NAMES,
           xlabel="Predicted soil class", ylabel="Reference soil class")
    ax.tick_params(axis="x", labelrotation=35)
    for y, x in np.ndindex(cm.shape):
        ax.text(x, y, f"{cm[y, x]}\n{100*fractions[y, x]:.1f}%", ha="center", va="center",
                color="white" if fractions[y, x] > .6 else "black", fontsize=8)
    fig.colorbar(heat, ax=ax, label="Row percentage")
    save_plot(fig, directory, 11)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    x = np.arange(6)
    for label, key, ax in zip(("Precision", "Recall", "F1"),
                               ("per_class_precision", "per_class_recall", "per_class_f1"), axes):
        for j, branch in enumerate(("FALL", "FS")):
            ax.bar(x+(j-.5)*.35, 100*np.array(metrics[branch][key]), width=.35, label=branch)
        ax.set(xticks=x, xticklabels=CLASS_NAMES, title=label, ylim=(0, 100))
        ax.tick_params(axis="x", labelrotation=55, labelsize=8)
    axes[0].set_ylabel("Score (%)")
    axes[-1].legend()
    save_plot(fig, directory, 12)


def main():
    identity = verify_identity()
    if readj(ROOT / "qa/result_acceptance.json")["run_identity_sha256"] != identity:
        raise ValueError("Reporting inputs belong to a different training run")
    directory = ROOT / "paper_outputs"
    directory.mkdir(exist_ok=True)
    selected = readj(ROOT / "results/selected_pipelines.json")
    metrics = {row["branch"]: row for row in readj(ROOT / "results/all_metrics.json") if row["policy"] == "R"}
    paired = pd.read_csv(ROOT / "analysis/paired_metric_differences.csv")
    fixed = []
    for family in FAMILIES:
        for outer in range(1, 6):
            cp = readj(checkpoint_path(["fusion", "R", outer, "FALL", family]))
            fixed.append(dict(classifier=family, outer_fold=outer, **{
                k: cp["outer_scores"][k] for k in METRIC_LABEL
            }))
    # Paper mode retains the frozen registry. Demo parameters come from the demo registry profile.
    from settings import protocol_options

    pool, _, _ = protocol_options()
    parameter_tables(directory, pool)
    backbone = selected_tables(directory, pool, selected)
    output_table(paired[paired.comparison.str.startswith("virtual_")], directory, "Table_6")
    alt = pd.read_csv(ROOT / "pi_sensitivity/threshold_performance_summary.csv")
    if readj(ROOT / "pi_sensitivity/acceptance.json")["parent_run_identity_sha256"] != identity:
        raise ValueError("Sensitivity runs belong to a different training run")
    baseline = {"cutoff": .001, "changed_prediction_count": 0, **{k: metrics["FS"][k] for k in METRIC_LABEL}}
    table7 = pd.concat([pd.DataFrame([baseline]), alt], ignore_index=True).sort_values("cutoff")
    for k in METRIC_LABEL:
        table7[k+"_delta_pp"] = 100*(table7[k] - baseline[k])
    output_table(table7, directory, "Table_7")
    output_table(fixed, directory, "Fig_10_fixed_classifier_folds")
    output_table(pd.DataFrame(metrics.values()).loc[:, ["branch", *METRIC_LABEL, "kappa"]],
                 directory, "Main_R_scores")
    meta = pd.read_csv(INPUT / "metadata.csv")
    plot_maps(directory, pd.read_csv(ROOT / "spatial/sample_groups.csv"))
    plot_workflow(directory)
    plot_spectra(directory, meta)
    plot_performance(directory, metrics, paired, fixed, backbone)
    print(f"{MODE} tables and review plots saved. Maps contain private locations on real runs.")
