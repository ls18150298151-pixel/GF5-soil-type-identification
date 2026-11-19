"""Check the local release allow-list, data-free tables, syntax and imports."""

import argparse
import ast
import csv
import importlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
TOP = {".gitignore", "README.md", "CODE_MAP.md", "DATA_AVAILABILITY.md",
       "CHANGES_FOR_RELEASE.md", "VERIFICATION.md", "requirements.txt",
       "config.example.json", "config.rasters.example.json"}
CODE = {"accept_PI_sensitivity", "aggregate", "cars_adapter", "environment_cleaner",
        "feature_algorithms", "formal_runner", "paired_analysis", "paper_outputs",
        "preflight", "prepare", "run", "run_PI_sensitivity", "settings",
        "spatial_folds", "spectral_preprocessing", "synthetic", "training_contract", "v2_features"}
PROTOCOL = {"candidate_pool_36.json", "factor_catalog_63.csv", "module_combinations_32.json",
            "PI_sensitivity_registration.json", "S1_factor_definitions.csv",
            "spectral_candidate_registry_972.json", "training_protocol.json"}
TOOLS = {"audit_release.py", "build_glo30_factors.py", "import_frozen.py"}
PATTERNS = {
    "absolute drive or account path": re.compile(r"(?i)(?:\b[a-z]:[/\\]|/(?:home|Users)/[\w.-]+|\\\\[\w.-]+\\[\w.-]+)"),
    "credential material": re.compile(r"(?i)(?:gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|sk-(?:proj-)?[\w-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer\s+[A-Za-z0-9._~-]{24,})"),
    "credential assignment": re.compile(r"""(?im)\b(?:api[_ -]?key|access[_ -]?token|password|cookie|client_secret)\s*[:=]\s*["'][^"'\r\n]{6,}["']"""),
}


def json_file(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def csv_file(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        return next(reader), list(reader)


def audit_structure():
    paths = TOP | {f"code/{n}.py" for n in CODE}
    paths |= {f"protocol/{n}" for n in PROTOCOL} | {f"tools/{n}" for n in TOOLS}
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
    missing = paths - actual - {"VERIFICATION.md"}
    extra = actual - paths
    if missing or extra or any(p.is_symlink() for p in ROOT.rglob("*")):
        raise SystemExit(f"Release allow-list failed. Missing: {sorted(missing)}; extra: {sorted(extra)}")
    count = 0
    for name in sorted(actual):
        source = (ROOT / name).read_text(encoding="utf-8")
        if name.endswith(".py"):
            ast.parse(source, filename=name)
            count += 1
        if name.endswith(".json"):
            json.loads(source)
        for label, pattern in PATTERNS.items():
            if pattern.search(source):
                raise SystemExit(f"{label} matched in {name}")
    pool = json_file("protocol/candidate_pool_36.json")["pool"]
    spectra = json_file("protocol/spectral_candidate_registry_972.json")["candidates"]
    combos = json_file("protocol/module_combinations_32.json")["combinations"]
    names = {r["candidate_id"]: r["classifier"] for r in pool}
    families = {(r["block"], r["reducer"], names[r["model_candidate_id"]]) for r in spectra}
    assert (len(pool), len(names), len(spectra), len(families), len(combos)) == (36, 36, 972, 48, 32)
    assert len({r["spectral_candidate_id"] for r in spectra}) == 972
    header, facts = csv_file("protocol/factor_catalog_63.csv")
    assert header == ["factor", "module", "kind"] and len(facts) == 63
    assert len({r[0] for r in facts}) == 63 and sum(r[1] != "E" for r in facts) == 59
    heading, table = csv_file("protocol/S1_factor_definitions.csv")
    assert len(heading) == 7 and all(len(row) == 7 for row in table)
    assert [int(row[0]) for row in table] == list(range(1, 64))
    for name in ("config.example.json", "config.rasters.example.json"):
        item = json_file(name)
        assert item["profile"] == "paper" and item["inputs"]["projected_crs"] == "EPSG:32648"
    print(f"PASS: {len(actual)} allowed files, {count} Python syntax checks, registries and text patterns")


def audit_imports():
    imports = {"scikit-learn": "sklearn"}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        importlib.import_module(imports.get(name, name))
        observed = metadata.version(name)
        print(f"{name}: import OK, installed {observed}, pinned {expected}")
    sys.path.insert(0, str(ROOT / "code"))
    for name in sorted(CODE):
        importlib.import_module(name)
    for name in ("build_glo30_factors", "import_frozen"):
        importlib.import_module(name)
    print(f"PASS: {len(CODE)+2} local modules imported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", action="store_true")
    args = parser.parse_args()
    audit_structure()
    if args.imports:
        audit_imports()
