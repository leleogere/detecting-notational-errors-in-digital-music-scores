import json
import logging
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Literal

import click

from contextual_errors_checker import check_path as check_path_contextual
from individual_errors_checker import check_path as check_path_individual
from cli_utils import check_paths_parallelizer
from cli_utils import display_statistics
from cli_utils import expand_paths

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@click.group("Notational error detector")
def cli():
    pass


@cli.command()
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["individual", "contextual", "both"]),
    required=True,
    help="Error checking mode. 'both' will first check individual errors, then contextual ones for the remaining scores.",
)
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    multiple=True,
    help="Either a path to a MusicXML file, a directory containing MusicXML files, or a text file containing the paths of MusicXML files to check (one per line). You can also pass this option multiple times for multiple paths.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("error_check_results.json"),
    help="Path to the output JSON file where the results will be saved.",
)
@click.option(
    "--njobs",
    "-j",
    type=int,
    default=1,
    help="Number of files to check in parallel. Use -1 for all available CPUs. Might crash in 'contextual' (and therefore 'both') mode, as Partitura scores sometimes has issues with Pickle.",
)
@click.option(
    "--statistics",
    "-s",
    is_flag=True,
    help="If set, display some statistics about the generated file.",
)
# fmt: on
def check(
    mode: Literal["individual", "contextual", "both"],
    path: Path | list[Path],
    output: Path,
    njobs: int,
    statistics: bool,
) -> None:
    """
    Check MusicXML files for errors.
    """
    # Expand paths
    paths = expand_paths(path)

    # Run the error detector
    all_exceptions = defaultdict(dict)
    if mode == "both":
        logger.info(f"Checking sequentially individual and contextual errors on {len(paths)} paths")
    if mode in ("individual", "both"):
        logger.info(f"Running individual error checker on {len(paths)} paths")
        # Check individual errors
        individual_exceptions = check_paths_parallelizer(
            check_path_individual,
            paths=paths,
            n_jobs=njobs,
            progress_bar=True,
            progress_bar_name="Checking individual errors",
        )
        # Filter out paths without exceptions
        individual_exceptions = {p: excs for p, excs in individual_exceptions.items() if excs}
        logger.info(
            f"Individual error checking completed: "
            f"{len(individual_exceptions)}/{len(paths)} files had errors"
        )
        # Add to all exceptions
        if individual_exceptions:
            all_exceptions["individual"].update(individual_exceptions)

    if mode == "both":
        total_paths = len(paths)
        # Filter out paths that had individual errors
        paths = [p for p in paths if p not in individual_exceptions]
        logger.info(f"{len(paths)} paths remain for contextual error checking")

    if mode in ("contextual", "both"):
        logger.info(f"Running contextual error checker on {len(paths)} paths")
        contextual_exceptions = check_paths_parallelizer(
            partial(
                check_path_contextual,
                allow_longer_measures=False,
                allow_duplicate_notes_in_chord=False,
            ),
            paths=paths,
            n_jobs=njobs,
            progress_bar=True,
            progress_bar_name="Checking contextual errors",
        )
        # Filter out paths without exceptions
        contextual_exceptions = {p: excs for p, excs in contextual_exceptions.items() if excs}
        logger.info(
            f"Contextual error checking completed: "
            f"{len(contextual_exceptions)}/{len(paths)} files had errors"
        )
        if contextual_exceptions:
            all_exceptions["contextual"].update(contextual_exceptions)

    if mode == "both":
        num_incorrect_paths = sum(len(p) for p in all_exceptions.values())
        logger.info(f"Both checks completed: {num_incorrect_paths}/{total_paths} files had errors")

    # Save results to output JSON file
    if all_exceptions:
        # Convert Path objects to strings for JSON serialization
        serializable_exceptions = {
            mode: {
                path.as_posix(): {
                    measure: [str(e) for e in measure_exceptions]
                    for measure, measure_exceptions in path_exceptions.items()
                }
                for path, path_exceptions in mode_exceptions.items()
            }
            for mode, mode_exceptions in all_exceptions.items()
        }
        with output.open("w") as f:
            json.dump(serializable_exceptions, f, indent=4)
        logger.info(f"Results saved to {output}")
        if statistics:
            display_statistics(output)
    else:
        logger.info("No errors found, no output file created")


@cli.command()
@click.argument(
    "error-file",
    type=click.Path(exists=True, path_type=Path),
)
def stats(error_file: Path) -> None:
    """
    Display some statistics about an error file.
    """
    display_statistics(error_file)


if __name__ == '__main__':
    cli()
