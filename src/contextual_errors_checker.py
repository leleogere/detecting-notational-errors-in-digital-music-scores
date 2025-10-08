import warnings
from collections import defaultdict
from pathlib import Path

import partitura as pt
import partitura.score

import contextual_errors_utils as tk


def check_automata(
    tokens: list[tk.Token],
    allow_longer_measures: bool,
    allow_duplicate_notes_in_chord: bool,
) -> dict[str, list[Exception]]:
    automata = tk.ConstrainedAutomata(
        num_voices=16,
        allow_longer_measures=allow_longer_measures,
        allow_duplicate_notes_in_chord=allow_duplicate_notes_in_chord,
    )

    exceptions = defaultdict(list)

    measure_number = None
    malformed_measure = False

    for token in tokens:
        if isinstance(token, tk.Bar):
            measure_number = token._partitura_elements[0].name
            malformed_measure = False

        # If malformed, skip until next Bar token
        if not malformed_measure:
            try:
                automata.transition(token)
            except tk.TransitionError as e:
                exceptions[measure_number].append(e)

                # Reset the automata
                malformed_measure = True
                automata.state.reset_to_clean_state()

    return exceptions


class TokenizationError(Exception):
    pass


def check_path(
    path: Path,
    allow_longer_measures: bool,
    allow_duplicate_notes_in_chord: bool,
) -> dict[str, list[Exception]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = pt.load_musicxml(path, force_note_ids=True)
        if len(score.parts) != 1:
            return {"global": [ValueError(f"More than 1 part: {len(score.parts)}")]}
        score = score.parts[0]
        score = pt.score.unfold_part_maximal(score, ignore_leaps=False)

    tk.fill_rests(score)
    try:
        tokens = tk.partitura_to_token_sequence(score)
    except Exception as e:
        return {"global": [e]}
    try:
        exceptions = check_automata(
            tokens,
            allow_longer_measures=allow_longer_measures,
            allow_duplicate_notes_in_chord=allow_duplicate_notes_in_chord,
        )
    except Exception as e:
        exceptions = {"global": [e]}

    return exceptions
