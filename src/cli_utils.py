import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
import logging

import joblib
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
MUSICXML_EXTENSIONS = {".xml", ".musicxml", ".mxl"}


def check_paths_parallelizer(
    fn: Callable[[Path], Any],
    paths: list[Path],
    n_jobs: int = 1,
    progress_bar: bool = True,
    progress_bar_name: str | None = None,
) -> dict[Path, Any]:
    """
    Parallelize a function over a list of paths, with optional progress bar.

    :param fn: Function to apply to each path.
    :param paths: List of paths to process.
    :param n_jobs: Number of parallel jobs. If 1, runs sequentially.
    :param progress_bar: Whether to show a progress bar.
    :param progress_bar_name: Name to display on the progress bar.
    :return: Dictionary mapping each path to its result.
    """
    tqdm_ = partial(tqdm, disable=not progress_bar, desc=progress_bar_name, total=len(paths))
    if n_jobs == 1:
        results = {p: fn(p) for p in tqdm_(paths)}
    else:
        results_list = joblib.Parallel(n_jobs=n_jobs, backend="threading", return_as="generator")(
            joblib.delayed(fn)(p) for p in paths
        )
        results = {p: r for p, r in tqdm_(zip(paths, results_list))}
    return results


def display_statistics(error_file: Path) -> None:
    """
    Display some statistics about an error file.

    :param error_file: Path to the error JSON file.
    """
    with error_file.open("r") as f:
        all_exceptions = json.load(f)

    for mode, exceptions_per_path in all_exceptions.items():
        if mode not in ("individual", "contextual"):
            raise ValueError(f"Unknown mode {mode} in error file {error_file}")

        total_measures_per_path = {}
        for path in exceptions_per_path:
            with open(path, "r") as f:
                total_measures_per_path[path] = f.read().count("<measure")

        n_paths_with_errors = len(exceptions_per_path)
        global_errors = [
            path for path, measures in exceptions_per_path.items() if "global" in measures
        ]
        n_measures_with_errors = sum(
            len(measures)
            for path, measures in exceptions_per_path.items()
            if path not in "global" not in measures  # Exclude global errors for per-measure stats
        )
        median_measure_error_per_score = sorted(
            len(measures)
            for path, measures in exceptions_per_path.items()
            if path not in global_errors
        )[n_paths_with_errors // 2]
        print(f"Mode: {mode}")
        if global_errors:
            print(
                f" - Number of paths with errors: {n_paths_with_errors} "
                f"(including {len(global_errors)} scores with a global error)"
            )
            print(
                f" - Number of measures with errors: {n_measures_with_errors} "
                f"(excluding scores with a global error)"
            )
        else:
            print(f" - Number of paths with errors: {n_paths_with_errors}")
            print(f" - Number of measures with errors: {n_measures_with_errors}")
        print(f" - Median number of flagged measures per score: {median_measure_error_per_score}")


def expand_paths(path: Path | list[Path]) -> list[Path]:
    all_paths = []
    if isinstance(path, Path):
        path = [path]

    # For all passed paths, check if they are files, directories, or text files
    for p in path:
        if not p.exists():
            logger.warning(f"Path {p} does not exist")
            continue
        # Simple case, a MusicXML file
        if p.suffix in MUSICXML_EXTENSIONS and p.is_file():
            logger.debug(f"New file to check: {p}")
            all_paths.append(p)
        # A directory, search recursively for MusicXML files
        elif p.is_dir():
            logger.debug(f"Expanding directory {p}:")
            for ext in MUSICXML_EXTENSIONS:
                new_files = list(p.rglob(f"*{ext}"))
                all_paths.extend(new_files)
                for file in new_files:
                    logger.debug(f"  - {file}")
        # A text file containing paths
        elif p.suffix in [".txt", ".csv", ".tsv"]:
            logger.debug(f"Reading text file {p}:")
            with p.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    line_path = Path(line)
                    if not line_path.is_absolute():
                        line_path = p.parent / line_path
                    if line_path.suffix in MUSICXML_EXTENSIONS and line_path.is_file():
                        all_paths.append(line_path)
                        logger.debug(f"  - {line_path}")
        else:
            logger.warning(f"Path {p} is not a MusicXML file, directory, or text file. Skipping")

    return sorted(set(all_paths))
