"""Paths and frozen protocol options for a local run."""

import json
import os
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
PROTOCOL = PACKAGE / "protocol"
CONFIG_PATH = Path(os.environ.get("GF5_CONFIG", PACKAGE / "config.example.json")).resolve()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
MODE = CONFIG["profile"]
if MODE not in {"paper", "demo"}:
    raise ValueError("profile must be paper or demo")


def resolve_path(value):
    return (CONFIG_PATH.parent / value).resolve()


ROOT = resolve_path(CONFIG["output_dir"])
if ROOT == PACKAGE or ROOT in PACKAGE.parents:
    raise ValueError("Use a dedicated run directory outside the source directories.")
if ROOT.is_relative_to(PACKAGE) and (
    not ROOT.is_relative_to(PACKAGE / "runs") or ROOT == PACKAGE / "runs"
):
    raise ValueError("Use a separate run directory or a named subdirectory under runs/.")
INPUT = ROOT / "prepared"
RULES = ROOT

CLASS_NAMES = ("red soil", "purple soil", "dry red soil", "yellow-brown soil",
               "limestone soil", "yellow soil")
CLASS_NAMES_ZH = ("红壤", "紫色土", "燥红土", "黄棕壤", "石灰岩土", "黄壤")


def protocol_options():
    pool = json.loads((PROTOCOL / "candidate_pool_36.json").read_text(encoding="utf-8"))["pool"]
    registry = json.loads((PROTOCOL / "spectral_candidate_registry_972.json").read_text(encoding="utf-8"))
    candidates, methods = registry["candidates"], registry["method_params"]
    if MODE == "paper":
        return pool, candidates, methods

    # Keep all families and branches, with deliberately smaller candidate grids.
    references = {"LightGBM-001", "XGBoost-016", "SVM-031", "RandomForest-043"}
    pool = [item for item in pool if item["candidate_id"] in references]
    for item in pool:
        params = item["params"]
        if "n_estimators" in params:
            params["n_estimators"] = 4
        if "min_child_samples" in params:
            params["min_child_samples"] = 4
        if item["classifier"] == "XGBoost":
            params["tree_method"] = "hist"
    reduced = []
    for item in candidates:
        method = item["reducer"]
        params = dict(methods[method][0])
        if item["model_candidate_id"] not in references or item["reducer_params"] != params:
            continue
        if method == "CARS":
            params["n_iterations"] = 3
        reduced.append(dict(item, reducer_params=params))
    methods = {method: [next(item["reducer_params"] for item in reduced if item["reducer"] == method)]
               for method in methods}
    return pool, reduced, methods
