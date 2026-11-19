"""Command line for the released analysis."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a local JSON configuration")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--phase", default="all",
                        choices=("all", "R_spectral", "R_fusion", "RV_spectral", "RV_fusion"))
    parser.add_argument("command", choices=("make-demo", "prepare", "split", "preflight", "train",
                                             "aggregate", "analyze", "pi-train", "pi-aggregate", "report", "all"))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    os.environ["GF5_CONFIG"] = str(Path(args.config).resolve())
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    if args.command == "make-demo":
        from synthetic import generate

        generate(Path(args.config).resolve())
        return

    from settings import ROOT

    marker = ROOT / ".gf5-run"
    if ROOT.exists() and any(ROOT.iterdir()) and not marker.is_file():
        raise ValueError("Run directory is not empty and was not created by this workflow.")
    ROOT.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)
    commands = (args.command,) if args.command != "all" else (
        "prepare", "split", "preflight", "train", "aggregate", "analyze",
        "pi-train", "pi-aggregate", "report",
    )
    for command in commands:
        if command == "prepare":
            from prepare import main as execute

            execute()
        elif command == "split":
            from spatial_folds import main as execute

            if (ROOT / "run_identity.json").exists():
                raise FileExistsError("This run is already frozen. Use a new output_dir to split again.")
            execute()
        elif command == "preflight":
            from preflight import main as execute

            execute(args.workers)
        elif command == "train":
            from formal_runner import main as execute

            execute(args.workers, "all" if args.command == "all" else args.phase)
        elif command == "aggregate":
            from aggregate import main as execute

            execute()
        elif command == "analyze":
            from paired_analysis import main as execute

            execute()
        elif command == "pi-train":
            from run_PI_sensitivity import main as execute

            execute(args.workers)
        elif command == "pi-aggregate":
            from accept_PI_sensitivity import main as execute

            execute()
        elif command == "report":
            from paper_outputs import main as execute

            execute()


if __name__ == "__main__":
    main()
