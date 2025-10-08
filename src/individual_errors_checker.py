import logging
import zipfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import joblib
from lxml import etree
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NOTE_TYPES = ["whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th", "256th"]
NOTE_TYPE_TO_QUARTER_LENGTH = {
    ntype: Fraction(1, 2) ** i for i, ntype in enumerate(NOTE_TYPES, start=-2)
}


class ParsingException(Exception):
    def __init__(self, msg: str, log_prefix: str | None = None):
        if log_prefix is None:
            log_prefix = ""
        logger.debug(f"{log_prefix}ParsingException: {msg}")
        super().__init__(msg)


def theoretical_note_duration_in_ql(note_type: str, dots: int = 0) -> Fraction:
    """
    Compute the theoretical duration of a note (eventually dotted) in quarter lengths.
    :param note_type: one of NOTE_TYPES
    :param dots: number of dots
    :return: duration in quarter lengths (as a Fraction object)
    """
    return NOTE_TYPE_TO_QUARTER_LENGTH[note_type] * (2 - Fraction(1, 2**dots))


def compute_theoretical_note_element_duration_ql(note: etree._Element) -> Fraction:
    """
    Compute the theoretical note duration in quarter lengths of a <note> MusicXML element.
    :param note: the <note> element
    :return: duration in quarter lengths (as a Fraction object)
    """
    if note.tag != "note":
        raise ValueError("The provided element is not a <note> element.")

    if (type_ := note.find("type")) is None:
        raise ValueError("The provided <note> element has no <type> child.")

    quarter_length = theoretical_note_duration_in_ql(
        note_type=type_.text,
        dots=len(note.findall("dot")),
    )

    # Take into account eventual tuplet
    if (time_modification := note.find("time-modification")) is not None:
        time_modification_ratio = compute_time_modification_ratio(time_modification)
        quarter_length *= time_modification_ratio

    return quarter_length


def split_composite_duration(dur: Fraction) -> list[tuple[str, Fraction]]:
    """
    Split a duration (in quarter length) into a list of atomic valid durations.
    Used to check if a <backup> or <forward> duration is valid.
    :param dur: the duration to split (in quarter notes)
    :return: a list of atomic parts as tuples (note_type, associated_duration)
    """
    remaining_duration = dur
    types_and_durations = iter(
        sorted(NOTE_TYPE_TO_QUARTER_LENGTH.items(), key=lambda x: x[1], reverse=True)
    )
    current_type, current_duration = next(types_and_durations)
    atomic_parts = []
    while remaining_duration > 0:
        if current_duration <= remaining_duration:
            remaining_duration -= current_duration
            atomic_parts.append((current_type, current_duration))
        else:
            try:
                current_type, current_duration = next(types_and_durations)
            except StopIteration:
                raise ValueError(f"Duration {dur} not decomposable.")
    return atomic_parts


def compute_time_modification_ratio(time_modification: etree._Element) -> Fraction:
    """
    Compute the time modification ratio from a <time-modification> MusicXML element.
    A note in an actual:normal tuplet has a duration of normal/actual * raw_duration.
    :param time_modification: the <time-modification> element
    :return: the ratio normal/actual as a Fraction object
    """
    actual_notes = int(time_modification.find("actual-notes").text)
    normal_notes = int(time_modification.find("normal-notes").text)
    return Fraction(normal_notes, actual_notes)


def check_divisions(
    xml: etree._ElementTree,
    log_prefix: str | None = None,
) -> dict[str, list[ParsingException]]:
    """
    Check that <divisions> tags in the MusicXML file are coherent with the theoretical durations of
    notes.
    :param xml: the MusicXML file as an lxml ElementTree
    :param log_prefix: a prefix to add to the log messages
    :return: a dictionary mapping measure numbers to lists of ParsingException instances
    """
    root = xml.getroot()

    exceptions = defaultdict(list)

    divisions = None
    for measure in root.iter('measure'):
        measure_number = measure.attrib["number"]

        # Look for divisions
        if (attributes := measure.find('attributes')) is not None and (  # Check if has attributes
            divisions_el := attributes.find("divisions")  # Check if has divisions
        ) is not None:
            divisions = int(divisions_el.text)
        if divisions is None:
            exceptions[measure_number].append(
                ParsingException(
                    msg=f"No divisions found in the first measure's attributes.",
                    log_prefix=log_prefix,
                )
            )
            break

        for element in measure:

            # If it's a note, check that its duration is coherent
            if element.tag == "note":

                # Skip grace notes
                if element.find("grace") is not None:
                    continue

                # Skip measure-long rests
                note_type_el = element.find("type")
                is_rest = element.find("rest") is not None
                if is_rest and note_type_el is None:
                    continue

                # Get actual duration in divisions
                actual_duration = int(element.find("duration").text)
                actual_quarter_length = Fraction(actual_duration, divisions)

                # Get some debugging information
                voice = int(element.find("voice").text) if element.find("voice") is not None else 1
                dots = len(element.findall("dot"))
                dur = note_type_el.text + "." * dots
                if is_rest:
                    note_info = f"rest/V{voice}/{dur}"
                else:
                    pitch_el = element.find("pitch")
                    if pitch_el is not None:
                        step = pitch_el.find("step").text
                        alter_el = pitch_el.find("alter")
                        octave = int(pitch_el.find("octave").text)
                        alter_int = 0 if alter_el is None else int(alter_el.text)
                        alter = "#" * alter_int + "#" * -alter_int
                        note_info = f"note/V{voice}/{dur}/{step}{alter}{octave}"
                    else:
                        note_info = f"note/V{voice}/{dur}"

                # Compute the theoretical duration of the note
                try:
                    theoretical_quarter_length = compute_theoretical_note_element_duration_ql(
                        element
                    )
                except KeyError:
                    exceptions[measure_number].append(
                        ParsingException(
                            msg=f"<{note_info}>: Unknown note type '{note_type_el.text}'",
                            log_prefix=log_prefix,
                        )
                    )
                    continue
                theoretical_duration = theoretical_quarter_length * divisions

                if theoretical_duration != actual_duration:
                    exceptions[measure_number].append(
                        ParsingException(
                            f"<{note_info}>: Actual and theoretical durations differ: "
                            f"{actual_duration} != {theoretical_duration} "
                            f"({actual_quarter_length} != {theoretical_quarter_length}) "
                            f"[{float(actual_quarter_length)} != "
                            f"{float(theoretical_quarter_length)}]",
                            log_prefix=log_prefix,
                        )
                    )

            if element.tag == "backup" or element.tag == "forward":
                # Check that the backup duration can be expressed as a combinations of valid times
                duration_ql = Fraction(int(element.find("duration").text), divisions)
                try:
                    split_composite_duration(duration_ql)
                except ValueError as e:
                    exceptions[measure_number].append(
                        ParsingException(
                            msg=f"<{element.tag}>: {e.args[0]}",
                            log_prefix=log_prefix,
                        )
                    )

    # Convert defaultdict to regular dict for return
    return dict(exceptions.items())


def check_path(path: Path) -> dict[str, list[ParsingException]]:
    """
    Check a MusicXML file for parsing exceptions related to divisions and note durations.
    :param path: the path to the MusicXML file
    :return: a dictionary mapping measure numbers to lists of ParsingException instances
    """
    # Allow to read compressed MusicXML files (.mxl)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zip_ref:
            with zip_ref.open("META-INF/container.xml") as container:
                container_tree: etree._ElementTree = etree.parse(container)
                rootfile_path = rootfile_path = container_tree.find(".//{*}rootfile").get(
                    "full-path"
                )
            with zip_ref.open(rootfile_path) as f:
                tree = etree.parse(f)
    else:
        tree = etree.parse(path)
    exceptions = check_divisions(tree, log_prefix=path.as_posix())
    return exceptions


def check_paths(
    paths: list[Path],
    n_jobs: int = 1,
) -> dict[Path, dict[str, list[ParsingException]]]:
    results = joblib.Parallel(n_jobs=n_jobs, return_as="generator")(
        joblib.delayed(check_path)(path) for path in paths
    )
    return {path: excs for path, excs in tqdm(zip(paths, results), total=len(paths)) if excs}
