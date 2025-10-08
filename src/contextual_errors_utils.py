import copy
import functools
import itertools
import operator
from collections import defaultdict
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Self
from typing import TypeVar
from typing import Unpack
from typing import assert_never

import partitura as pt
import partitura.score
from partitura.utils.globals import INT_TO_ALT


class SymbolicDurationType(StrEnum):
    LONG = "long", Fraction(16)
    BREVE = "breve", Fraction(8)
    WHOLE = "whole", Fraction(4)
    HALF = "half", Fraction(2)
    QUARTER = "quarter", Fraction(1)
    EIGHTH = "eighth", Fraction(1, 2)
    N16TH = "16th", Fraction(1, 4)
    N32ND = "32nd", Fraction(1, 8)
    N64TH = "64th", Fraction(1, 16)
    N128TH = "128th", Fraction(1, 32)
    N256TH = "256th", Fraction(1, 64)

    def __new__(cls, value: str, quarter_length: Fraction):
        obj = str.__new__(cls)
        obj._value_ = value
        obj.quarter_length = quarter_length
        return obj

    @classmethod
    def min(cls):
        return min(cls, key=operator.attrgetter("quarter_length"))

    @classmethod
    def max(cls):
        return max(cls, key=operator.attrgetter("quarter_length"))

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class SymbolicDuration:
    type: SymbolicDurationType
    dots: int = 0

    def __post_init__(self):
        if isinstance(self.type, str):
            object.__setattr__(self, "type", SymbolicDurationType(self.type))
        if self.dots > 3:
            raise ValueError("A note can't have more than 3 dots.")

    @functools.cached_property
    def quarter_length(self) -> Fraction:
        # The expression returned is obtained from the sum of a geometric progression
        # sum(
        #     (self.type.quarter_length * Fraction(1, 2) ** i for i in range(self.dots + 1)),
        #     Fraction(0),
        # )
        return self.type.quarter_length * (2 - Fraction(1, 2**self.dots))

    def __str__(self):
        return f"{self.type.value}{'.' * self.dots}"

    def __eq__(self, other):
        return self.type is other.type and self.dots == other.dots  # bc issues with == with types


NoteTuple = namedtuple("NoteTuple", ["name", "octave", "accidental"])
TieProperties = namedtuple("TieProperties", ["note_tuple", "min_resolution_pos"])


class TransitionCondition:
    AND = "𝗔𝗡𝗗"
    OR = "𝗢𝗥"
    NOT = "𝗡𝗢𝗧"

    def __init__(
        self,
        condition: Callable[[], bool],
        expr: str,
        alias=None,
        parent_operator: str | None = None,
    ):
        if not callable(condition):
            raise TypeError("Condition must be callable.")
        self.condition = condition
        self.expr = expr
        self.alias = alias
        self._children = []
        self._evaluated = None
        self.parent_operator = parent_operator

    @property
    def value(self) -> bool:
        if self._evaluated is None:
            self._evaluated = self.condition()
        return self._evaluated

    def with_alias(self, alias):
        self.alias = alias
        return self

    def with_prefix_alias(self, prefix: str):
        self.alias = prefix + self.alias if self.alias is not None else prefix + self.expr
        return self

    def __bool__(self):
        return self.value

    def __invert__(self) -> Self:
        def cond():
            return not self.value

        expr = f"{self.NOT}({self.expr})"
        alias = f"{self.NOT}({self.alias})" if self.alias else None
        tc = TransitionCondition(cond, expr=expr, alias=alias, parent_operator=self.NOT)
        tc._children = [self]
        return tc

    @classmethod
    def and_(
        cls, *all_conditions: Unpack[Self], empty_value: bool = True, skip_nones: bool = True
    ) -> Self:
        if skip_nones:
            all_conditions = list(filter(lambda c: c is not None, all_conditions))

        if not all_conditions:
            return cls(lambda: empty_value, str(empty_value))

        def cond():
            for c in all_conditions:
                if not c.value:
                    return False
            return True

        expr = f"  {cls.AND}  ".join(cls._wrap(c.alias or c.expr) for c in all_conditions)
        tc = TransitionCondition(cond, expr=expr, parent_operator=cls.AND)
        tc._children = list(all_conditions)
        return tc

    @classmethod
    def or_(
        cls, *all_conditions: Unpack[Self], empty_value: bool = True, skip_nones: bool = True
    ) -> Self:
        if skip_nones:
            all_conditions = list(filter(lambda c: c is not None, all_conditions))

        if not all_conditions:
            return cls(lambda: empty_value, str(empty_value))

        def cond():
            for c in all_conditions:
                if c.value:
                    return True
            return False

        expr = f"  {cls.OR}  ".join(cls._wrap(c.alias or c.expr) for c in all_conditions)
        tc = TransitionCondition(cond, expr=expr, parent_operator=cls.OR)
        tc._children = list(all_conditions)
        return tc

    @classmethod
    def _wrap(cls, label: str) -> str:
        if f" {cls.AND} " in label or f" {cls.OR} " in label:
            return f"({label})"
        return label

    def validate(
        self,
        token: "AnyToken | None" = None,
        state: "TranscriptionState | None" = None,
    ) -> None:
        """
        Raises a TransitionError if the condition is not met. The two additional parameters are only
        passed to the exception if it is raised.
        """
        if not self:
            raise TransitionError(
                logical_tree=self.format_tree(include_all_branches=False, include_check=True),
                token=token,
                state=state,
            )

    def _build_tree(
        self,
        include_all_branches: bool = False,
        _invert: bool = False,
    ) -> tuple[Self, list[tuple]]:
        if not self._children:
            return self, []
        if include_all_branches:
            children = [
                child._build_tree(include_all_branches=True, _invert=False)
                for child in self._children
            ]
        else:
            if not _invert:
                op_short_circuited = self.AND
                op_not_short_circuited = self.OR
                short_circuit_on = False
            else:
                op_short_circuited = self.OR
                op_not_short_circuited = self.AND
                short_circuit_on = True

            # If parent is AND, returns only the first error
            if self.parent_operator == op_short_circuited:
                for child in self._children:
                    if child.value == short_circuit_on:
                        children = [child._build_tree(_invert=_invert)]
                        break
                else:
                    raise ValueError(
                        "At least one child of this transition should have failed with `assert_failure`."
                    )

            # If parent is OR, returns all
            elif self.parent_operator == op_not_short_circuited:
                children = [
                    child._build_tree(_invert=_invert) for child in self._children if not child
                ]

            elif self.parent_operator == self.NOT:
                assert len(self._children) == 1
                children = [self._children[0]._build_tree(_invert=not _invert)]

            else:
                raise ValueError(f"Unknown parent operator: {self.parent_operator}")
        return self, children

    def _format_tree(self, node, prefix="", is_last=True, include_check: bool = False):
        cond, children = node
        if include_check:
            check = "✅ " if cond else "❌ "
        else:
            check = ""
        line = prefix + ("└── " if is_last else "├── ") + check
        if cond.alias is not None:
            line += f"{cond.alias}  🢂  {cond.expr}"
        else:
            line += cond.expr
        lines = [line]
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(children):
            last = i == len(children) - 1
            lines.append(
                self._format_tree(
                    node=child,
                    prefix=child_prefix,
                    is_last=last,
                    include_check=include_check,
                )
            )
        return "\n".join(lines)

    def format_tree(self, include_all_branches: bool = True, include_check: bool = False) -> str:
        tree = self._build_tree(include_all_branches=include_all_branches)
        return self._format_tree(tree, include_check=include_check)

    def print_tree(self, include_all_branches: bool = True, include_check: bool = False) -> None:
        print(
            self.format_tree(include_all_branches=include_all_branches, include_check=include_check)
        )

    def __repr__(self):
        components = []
        if self.alias is not None:
            components.append(self.alias)
        components.append(self.expr)
        return f"TransitionCondition({', '.join(components)})"


class ChordState:
    def __init__(self, symbolic_duration: SymbolicDuration):
        self._symbolic_duration: SymbolicDuration = symbolic_duration
        self.notes: list[NoteTuple] = []

    @property
    def duration(self) -> SymbolicDuration:
        return self._symbolic_duration

    @property
    def num_notes(self) -> int:
        """Returns the number of notes in the chord."""
        return len(self.notes)

    @property
    def can_resolve_condition(self) -> TransitionCondition:
        return TransitionCondition(lambda: self.num_notes >= 2, "n_notes > 2")

    def __str__(self):
        return f"Chord({self.duration}, {self.num_notes} notes)"


class SegmentCompletionState:
    """This class represent the completion state of an abstract segment (tuplet or measure)."""

    def __init__(self, duration: Fraction):
        self.duration = duration
        self.position = Fraction(0)

    @property
    def remaining_duration(self) -> Fraction | float:
        return self.duration - self.position

    @property
    def has_space_left_condition(self) -> TransitionCondition:
        return TransitionCondition(
            lambda: self.remaining_duration > 0, "rem_duration > 0", "space left"
        )

    def advance_position(self, increment: Fraction) -> None:
        """Increment the current position by a given positive duration.
        The increment must not exceed the total duration.
        """
        if increment < 0:
            raise ValueError("Increment must be positive.")
        if increment > self.remaining_duration:
            raise ValueError("Increment must not exceed the total duration.")
        self.position += increment


class TupletCompletionState(SegmentCompletionState):
    def __init__(self, n_actual: int, n_normal: int, unit_duration: SymbolicDuration):
        self.n_actual = n_actual
        self.n_normal = n_normal
        self.unit_duration = unit_duration
        super().__init__(self.actual_duration)

    @property
    def normal_duration(self) -> Fraction:
        """Returns the normal duration of the tuplet."""
        return self.n_normal * self.unit_duration.quarter_length

    @property
    def actual_duration(self) -> Fraction:
        """Returns the actual duration of the tuplet."""
        return self.n_actual * self.unit_duration.quarter_length

    @property
    def duration_multiplier(self) -> Fraction:
        return Fraction(self.n_normal, self.n_actual)

    def __str__(self):
        return f"Tuplet(pos={self.position}, rem={self.remaining_duration}, type={self.n_actual}:{self.n_normal} {self.unit_duration})"


class VoiceCompletionState(SegmentCompletionState):
    def __init__(self, total_time: Fraction):
        super().__init__(total_time)
        self.chord: ChordState | None = None
        self.tuplet: TupletCompletionState | None = None

    @property
    def in_chord_condition(self) -> TransitionCondition:
        return TransitionCondition(lambda: self.chord is not None, "in chord")

    @property
    def in_tuplet_condition(self) -> TransitionCondition:
        return TransitionCondition(lambda: self.tuplet is not None, "in tuplet")

    @property
    def in_structure_condition(self) -> TransitionCondition:
        return TransitionCondition.or_(
            self.in_chord_condition,
            self.in_tuplet_condition,
        )

    @property
    def completed_condition(self) -> TransitionCondition:
        return TransitionCondition.and_(
            ~self.in_structure_condition,
            ~self.has_space_left_condition,
        )

    @property
    def position_positive_condition(self) -> TransitionCondition:
        return TransitionCondition(lambda: self.position > 0, "position > 0")

    @property
    def started_condition(self) -> TransitionCondition:
        return TransitionCondition.or_(
            self.position_positive_condition,
            self.in_structure_condition,
        )

    @property
    def segment_remaining_duration(self) -> Fraction:
        """Returns the remaining time in the current segment (measure or tuplet)."""
        if self.in_tuplet_condition:
            return self.tuplet.remaining_duration
        else:
            return self.remaining_duration

    def advance_position(self, duration: Fraction) -> None:
        """Advance the current position by a given positive duration, taking into
        account an eventual tuplet (in that case will advance of the duration in
        the tuplet time referential)."""
        if self.in_tuplet_condition:
            self.tuplet.advance_position(duration)
            super().advance_position(duration * self.tuplet.duration_multiplier)
        else:
            super().advance_position(duration)

    def __str__(self):
        return f"VoiceCompletionState(pos={self.position}, tot={self.duration}, rem={self.remaining_duration}, chord={self.chord}, tuplet={self.tuplet})"


class TranscriptionState:
    def __init__(self, num_voices: int = 1, initial_measure_number: int = 1):
        if num_voices <= 1:
            raise ValueError("num_voices must be at least 1.")
        self._num_voices = num_voices
        self.voices: list[VoiceCompletionState] = []
        self.tied_notes: dict[int, TieProperties] = {}  # pitch -> [note_tuple, min_resolution_pos]
        self.measure_number = initial_measure_number

    @property
    def num_voices(self) -> int:
        return self._num_voices

    @property
    def voices_initialized_condition(self) -> TransitionCondition:
        return TransitionCondition(
            lambda: len(self.voices) > 0,
            "len(voices) > 0",
            "voices initialized",
        )

    def resolvable_tied_notes(self, position: Fraction) -> dict[int, TieProperties]:
        return {
            pitch: props
            for pitch, props in self.tied_notes.items()
            if props.min_resolution_pos <= position
        }

    def copy(self, deep: bool = True) -> Self:
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def check_for_all_voices_condition(
        self,
        condition: Callable[[VoiceCompletionState], TransitionCondition],
        filter_fn: Callable[[VoiceCompletionState], bool] | None = lambda _: True,
    ) -> TransitionCondition:
        return TransitionCondition.and_(
            self.voices_initialized_condition,
            TransitionCondition.and_(
                *(
                    condition(voice).with_prefix_alias(f"V{i}: ")
                    for i, voice in enumerate(self.voices)
                    if filter_fn(voice)
                )
            ),
        ).with_alias("check all voices")

    def check_for_any_voices_condition(
        self,
        condition: Callable[[VoiceCompletionState], TransitionCondition],
        filter_fn: Callable[[VoiceCompletionState], bool] | None = lambda _: True,
    ) -> TransitionCondition:
        return TransitionCondition.and_(
            self.voices_initialized_condition,
            TransitionCondition.or_(
                *(
                    condition(voice).with_prefix_alias(f"V{i}: ")
                    for i, voice in enumerate(self.voices)
                    if filter_fn(voice)
                )
            ),
        ).with_alias("check any voice")

    @property
    def are_started_voices_synced_and_not_in_structure_condition(self) -> TransitionCondition:
        all_positions = [v.position for v in self.voices]
        max_pos = max(all_positions, default=-1)
        return self.check_for_all_voices_condition(
            lambda v: TransitionCondition.and_(
                TransitionCondition(
                    lambda: v.current_position == max_pos,
                    f"pos = {max_pos}",
                    f"must end at max position",
                ),
                ~v.in_structure_condition,
            ),
            filter_fn=operator.attrgetter("started_condition"),
        ).with_alias("voices are in sync and not in structure")

    @property
    def are_started_voices_complete_condition(self) -> TransitionCondition:
        return TransitionCondition.and_(
            self.are_started_voices_synced_and_not_in_structure_condition,
            self.check_for_any_voices_condition(lambda v: v.completed_condition),
        ).with_alias("started voices are complete")

    def initialize_voices(self, total_time: Fraction) -> None:
        self.voices = [VoiceCompletionState(total_time) for _ in range(self.num_voices)]

    def reset_to_clean_state(self):
        """
        Useful when the sequence has errors and you want to reset the automata into a clean state
        (for example after the next Bar token).
        """
        self.voices = []
        self.tied_notes = {}

    def __str__(self):
        tied_notes_str = [
            f'{nt.name}{nt.octave}{INT_TO_ALT[nt.accidental]} ({p}): {pos}'
            for p, (nt, pos) in self.tied_notes.items()
        ]
        components = [
            f"measure={self.measure_number}",
            f"tied_notes={{{', '.join(tied_notes_str)}}}",
            *[f"voice{i}={v}" for i, v in enumerate(self.voices)],
        ]
        sep = ",\n  "
        return f"TranscriptionState(\n  {sep.join(components)}\n)"


@dataclass(kw_only=True)
class Token:
    def __str__(self):
        elements = [
            f"{k}={str(v)}"
            for k, v in self.__dict__.items()
            if not k.startswith("_") and v is not None
        ]
        return f"{self.__class__.__name__}({', '.join(elements)})"

    def __getstate__(self):
        # Remove eventual _partitura_elements attribute when pickling
        state = self.__dict__.copy()
        state.pop('_partitura_elements', None)
        return state

    @classmethod
    def as_str(cls) -> str:
        return cls.__name__


@dataclass
class EndOfScore(Token):
    pass


@dataclass
class Bar(Token):
    numerator: int
    denominator: int
    duration: Fraction | None = None

    def __post_init__(self):
        if self.denominator.bit_count() != 1:
            raise ValueError("The denominator of the TS must be a power of 2.")
        if self.numerator < 1 or self.denominator < 1:
            raise ValueError("All time signature attributes values must be positive integers.")
        if self.duration is None:
            self.duration = self.nominal_duration

    @property
    def nominal_duration(self) -> Fraction:
        """Compute the nominal duration of the measure in quarter notes, according to the time
        signature. Note that this is not necessarily the actual true duration of the measure,
        in case of anacrusis or incomplete measures for example.

        The unit duration (in quarter notes) is computed using some bit manipulation
        to find the closest power of 2 (x = int.bit_length() - 1), then the duration
        is computed using 2**(2 - x)
            - For denominator = 2 (half note): 2**(2 - (2 - 1)) = 2
            - For denominator = 4 (quarter note): 2**(2 - (3 - 1)) = 1
            - For denominator = 8 (eighth note): 2**(2 - (4 - 1)) = 1/2
        """
        denominator_duration = Fraction(2 ** (3 - self.denominator.bit_length()))
        return self.numerator * denominator_duration


@dataclass
class HasVoiceMixin:
    """Used for elements that are tied to a voice."""

    voice_idx: int

    @property
    def voice_idx_1b(self):
        """Returns the voice index but one-based."""
        return self.voice_idx + 1


@dataclass
class HasDurationMixin:
    """Used for elements that have a duration."""

    duration: SymbolicDuration


@dataclass
class Rest(Token, HasVoiceMixin, HasDurationMixin):
    is_measure: bool = False


@dataclass
class Note(Token, HasVoiceMixin, HasDurationMixin):
    note_class: str
    octave: int
    accidental: int
    tied: bool
    grace: bool = False

    _note_classes = "ABCDEFG"
    _note_alterations = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"}
    _octave_range = (0, 8)

    def __post_init__(self):
        if self.note_class not in self._note_classes:
            raise ValueError(f"The note class must be one of {', '.join(self._note_classes)}.")
        if self.octave < self._octave_range[0] or self.octave > self._octave_range[1]:
            raise ValueError(
                f"The octave must be between {self._octave_range[0]} and {self._octave_range[1]}."
            )
        if self.accidental not in self._note_alterations.keys():
            raise ValueError(
                f"The accidental must be one of {', '.join(map(str, self._note_alterations.keys()))}, got {self.accidental}."
            )

    @property
    def note_tuple(self) -> NoteTuple:
        return NoteTuple(name=self.note_class, octave=self.octave, accidental=self.accidental)

    @property
    def pitch(self) -> int:
        return pt.score.pitch_spelling_to_midi_pitch(
            step=self.note_class,
            alter=self.accidental,
            octave=self.octave,
        )


@dataclass
class ChordStart(Token, HasVoiceMixin, HasDurationMixin):
    pass


@dataclass
class ChordEnd(Token, HasVoiceMixin):
    pass


@dataclass
class TupletStart(Token, HasVoiceMixin):
    n_actual: int
    n_normal: int
    unit_duration: SymbolicDuration

    @property
    def actual_duration(self) -> Fraction:
        return self.n_actual * self.unit_duration.quarter_length

    @property
    def normal_duration(self) -> Fraction:
        return self.n_normal * self.unit_duration.quarter_length

    def is_token_allowed_condition(self, state: TranscriptionState) -> TransitionCondition:
        voice = state.voices[self.voice_idx]
        return TransitionCondition.and_(
            state.voices_initialized_condition,
            TransitionCondition.and_(
                ~voice.in_structure_condition,
                TransitionCondition(
                    lambda: self.normal_duration
                    <= state.voices[self.voice_idx].segment_remaining_duration,
                    "norm_dur <= rem_dur",
                    "tuplet fits in remaining duration",
                ),
            ),
        )

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.voices[self.voice_idx].tuplet = TupletCompletionState(
            n_actual=self.n_actual,
            n_normal=self.n_normal,
            unit_duration=self.unit_duration,
        )
        return state


@dataclass
class TupletEnd(Token, HasVoiceMixin):
    def is_token_allowed_condition(self, state: TranscriptionState) -> TransitionCondition:
        voice = state.voices[self.voice_idx]
        return TransitionCondition.and_(
            voice.in_tuplet_condition,
            TransitionCondition(
                lambda: voice.segment_remaining_duration == 0, "rem_dur = 0", "tuplet finished"
            ),
        )

    def _transition_fn(self, state: TranscriptionState) -> TranscriptionState:
        state.voices[self.voice_idx].tuplet = None
        return state


TOKEN_TYPES = [
    EndOfScore,
    Bar,
    Rest,
    Note,
    ChordStart,
    ChordEnd,
    TupletStart,
    TupletEnd,
]

TOKEN_TYPES_STR = [cls.__name__ for cls in TOKEN_TYPES]

TOKEN_STR_TO_TYPE = dict(zip(TOKEN_TYPES_STR, TOKEN_TYPES))


AnyToken = TypeVar("AnyToken", bound=Token)


class TransitionError(Exception):
    def __init__(
        self,
        logical_tree: str,
        token: AnyToken,
        state: TranscriptionState,
    ):
        super().__init__()
        self.logical_tree = logical_tree
        self.token = token
        self.state = state

    def __str__(self):
        msg = (
            f"TransitionError: cannot process token {self.token} in state:\n{self.state}\n"
            f"Logical tree:\n{self.logical_tree}"
        )
        return msg


def fill_rests(part: pt.score.Part) -> None:
    """
    Fill rests in incomplete voices of a part.

    :param part: Part to fill rests in
    """
    divisions = int(part.quarter_duration_map(0))

    def create_rests(
        duration: Fraction,
        largest_last: bool = False,
    ) -> list[tuple[pt.score.Rest, SymbolicDuration]]:
        # print(f"Creating rests for duration {duration}")
        position = Fraction(0)
        allowed_durations = iter(SymbolicDurationType)
        tested_duration = next(allowed_durations)
        rests: list[tuple[pt.score.Rest, SymbolicDuration]] = []
        while position < duration:
            # print(f"{position=} \t{duration=} \t{tested_duration=}")
            if position + tested_duration.quarter_length <= duration:
                # print(f"Adding rest of duration {tested_duration.value} at position {position}")
                rests.append(
                    (
                        pt.score.Rest(symbolic_duration={"type": tested_duration.value}),
                        tested_duration,
                    )
                )
                position += tested_duration.quarter_length
            else:
                try:
                    tested_duration = next(allowed_durations)
                except StopIteration:
                    raise ValueError(
                        f"Cannot fill rest of duration {duration} at position {position} in "
                        f"measure {measure.name} (voice {voice_idx})"
                    )
        if largest_last:
            rests = list(reversed(rests))
        return rests

    for measure in part.measures:
        start_time = measure.start.t
        end_time = measure.end.t
        notes_and_tuplets = list(
            filter(
                lambda n: not isinstance(n, pt.score.GraceNote),
                part.iter_all(
                    # [pt.score.GenericNote, pt.score.Tuplet],
                    pt.score.GenericNote,
                    start_time,
                    end_time,
                    include_subclasses=True,
                ),
            )
        )
        voices = defaultdict(list)
        for note_or_tuplet in notes_and_tuplets:
            idx = (
                note_or_tuplet.voice
                if isinstance(note_or_tuplet, pt.score.GenericNote)
                else note_or_tuplet.start_note.voice
            )
            voices[idx].append(note_or_tuplet)
        for voice_idx, voice_notes_and_tups in voices.items():
            time = start_time
            for note_or_tup in voice_notes_and_tups:
                if isinstance(note_or_tup, pt.score.Tuplet):
                    pass
                else:
                    note = note_or_tup
                    if note.start.t > time:
                        # print(
                        #     f"Voice {voice_idx} has a space from {time} to {note.start.t} "
                        #     f"(measure {measure.number}, from {Fraction(time - start_time, divisions)} "
                        #     f"to {Fraction(note.start.t - start_time, divisions)})"
                        # )
                        rests = create_rests(Fraction(note.start.t - time, divisions))
                        # print(rests)
                        for rest, dur in rests:
                            start = time
                            end = start + int(dur.quarter_length * divisions)
                            rest.voice = voice_idx
                            rest.staff = note.staff
                            part.add(rest, start=start, end=end)
                            time = end
                        assert end == note.start.t
                time = note_or_tup.end.t
            if time < end_time:
                # print(
                #     f"Voice {voice_idx} has a space from {time} to {end_time} "
                #     f"(measure {measure.number}, from {Fraction(time - start_time, divisions)} "
                #     f"to {Fraction(end_time - start_time, divisions)})"
                # )
                rests = create_rests(Fraction(end_time - time, divisions), largest_last=True)
                for rest, dur in rests:
                    start = time
                    end = time + int(dur.quarter_length * divisions)
                    rest.voice = voice_idx
                    rest.staff = note.staff
                    part.add(rest, start=start, end=end)
                    time = end


def partitura_to_token_sequence(score: pt.score.Part) -> list[AnyToken]:
    """Convert a Partitura part to a sequence of tokens.

    :param score: The Partitura part to convert.
    :return: A list of tokens (not necessarily valid, will need to be checked by an automata).
    """
    tokens: list[AnyToken] = []
    time_signature = None
    pending_tuplets: list[pt.score.Tuplet] = []

    for point in score._points:
        point: pt.score.TimePoint

        # Start by looking for eventual TS change
        if point.starting_objects[pt.score.TimeSignature]:
            time_signature = list(point.starting_objects[pt.score.TimeSignature].keys())[0]

        # Finish pending tuplets if needed
        for tuplet in reversed(pending_tuplets):
            if tuplet.end_note.end.t <= point.t:
                token = TupletEnd(
                    voice_idx=tuplet.end_note.voice - 1,  # 1-indexed
                )
                token._partitura_elements = [tuplet]
                tokens.append(token)
                pending_tuplets.remove(tuplet)

        # Look over different objet types, ORDER MATTERS

        # Look for TimeSignature
        time_signatures = list(point.starting_objects[pt.score.TimeSignature].keys())
        assert (
            len(time_signatures) <= 1
        ), "Multiple time signatures at the same time are not supported."
        if len(time_signatures) == 1:
            time_signature = time_signatures[0]

        # Look for Measure
        measures = list(point.starting_objects[pt.score.Measure].keys())
        if len(measures) > 1:
            raise ValueError(f"Simultaneous bars are not supported, got {len(measures)} measures.")
        elif len(measures) == 1:
            measure = measures[0]
            real_duration_in_divs = measure.end.t - measure.start.t
            real_duration_in_quarters = Fraction(real_duration_in_divs, point.quarter)
            measure_token = Bar(
                numerator=time_signature.beats,
                denominator=time_signature.beat_type,
                duration=real_duration_in_quarters,
            )
            measure_token._partitura_elements = [measure, time_signature]
            tokens.append(measure_token)

        # Look for Tuplet
        tuplets = sorted(
            point.starting_objects[pt.score.Tuplet].keys(),
            key=lambda t: t.start_note.voice,
        )
        for tuplet in tuplets:
            # Weird case where we encounter things like 6:6 tuplets (which shouldn't be a tuplet)
            if (
                tuplet.actual_type == tuplet.normal_type
                and tuplet.actual_dots == tuplet.normal_dots
                and tuplet.actual_notes == tuplet.normal_notes
            ):
                continue
            # Deal with the rare case of different tuplet types
            if tuplet.actual_type != tuplet.normal_type or tuplet.actual_dots != tuplet.normal_dots:
                durations_ratio = tuplet.duration_multiplier
                normal_notes = tuplet.actual_notes * durations_ratio
                assert (
                    normal_notes.denominator == 1
                ), f"Tuplet type conversion not possible: {tuplet.actual_notes}-{tuplet.actual_type} against {tuplet.normal_notes}-{tuplet.normal_type}."
                normal_notes = int(normal_notes)
            else:
                normal_notes = tuplet.normal_notes
            token = TupletStart(
                voice_idx=tuplet.start_note.voice - 1,  # 1-indexed
                n_actual=tuplet.actual_notes,
                n_normal=normal_notes,
                unit_duration=SymbolicDuration(
                    type=SymbolicDurationType(tuplet.actual_type),
                    dots=tuplet.actual_dots,
                ),
            )
            token._partitura_elements = [tuplet]
            tokens.append(token)
            pending_tuplets.append(tuplet)

        # Look for notes and grace note
        notes = sorted(
            itertools.chain(
                point.starting_objects[pt.score.GraceNote].keys(),  # Grace notes before notes
                point.starting_objects[pt.score.Note].keys(),
            ),
            key=lambda n: (isinstance(n, pt.score.GraceNote), n.voice, n.midi_pitch, n.step),
        )
        # Get notes per voice
        notes_per_voice = defaultdict(lambda: {"grace": [], "note": []})
        for note in notes:
            type_ = "grace" if isinstance(note, pt.score.GraceNote) else "note"
            notes_per_voice[note.voice][type_].append(note)
        for voice, vnotes in notes_per_voice.items():
            gnotes, nnotes = vnotes.values()
            # Add grace notes
            for gnote in gnotes:
                token = Note(
                    note_class=gnote.step,
                    octave=gnote.octave,
                    accidental=gnote.alter or 0,
                    duration=SymbolicDuration(type=SymbolicDurationType.EIGHTH),
                    tied=False,
                    voice_idx=gnote.voice - 1,
                    grace=True,
                )
                token._partitura_elements = [gnote]
                tokens.append(token)
            if len(nnotes) > 1:
                token = ChordStart(
                    voice_idx=voice - 1,
                    duration=SymbolicDuration(
                        type=SymbolicDurationType(nnotes[0].symbolic_duration["type"]),
                        dots=(
                            nnotes[0].symbolic_duration["dots"]
                            if "dots" in nnotes[0].symbolic_duration
                            else 0
                        ),
                    ),
                )
                token._partitura_elements = nnotes
                tokens.append(token)
            for note in nnotes:
                token = Note(
                    note_class=note.step,
                    octave=note.octave,
                    accidental=note.alter or 0,
                    duration=SymbolicDuration(
                        type=SymbolicDurationType(note.symbolic_duration["type"]),
                        dots=(
                            note.symbolic_duration["dots"]
                            if "dots" in note.symbolic_duration
                            else 0
                        ),
                    ),
                    tied=note.tie_next is not None,
                    voice_idx=note.voice - 1,
                )
                for attr in note.articulations + note.ornaments:
                    if hasattr(token, attr_ := attr.replace("-", "_")):
                        setattr(token, attr_, True)
                token._partitura_elements = [note]
                tokens.append(token)
            if len(nnotes) > 1:
                token = ChordEnd(voice_idx=voice - 1)  # 1-indexed
                token._partitura_elements = nnotes
                tokens.append(token)

        # Look for Rest
        rests = sorted(
            point.starting_objects[pt.score.Rest].keys(),
            key=operator.attrgetter("voice"),
        )
        for rest in rests:
            is_measure = "type" not in rest.symbolic_duration
            rest_type = rest.symbolic_duration.get("type", "whole")  # default whole for bar-rests
            token = Rest(
                duration=SymbolicDuration(
                    type=SymbolicDurationType(rest_type),
                    dots=(
                        rest.symbolic_duration["dots"] if "dots" in rest.symbolic_duration else 0
                    ),
                ),
                voice_idx=rest.voice - 1,  # 1-indexed
                is_measure=is_measure,
            )
            token._partitura_elements = [rest]
            tokens.append(token)

    tokens.append(EndOfScore())

    return tokens


class ConstrainedAutomata:
    def __init__(
        self,
        num_voices: int = 1,
        allow_duplicate_notes_in_chord: bool = True,
        allow_longer_measures: bool = False,
    ):
        self.state = TranscriptionState(num_voices=num_voices)
        self.allow_duplicate_notes_in_chord = allow_duplicate_notes_in_chord
        self.allow_longer_measures = allow_longer_measures

    def can_transition(self, token: Token) -> bool:
        return bool(self.is_token_allowed_condition(token))

    def transition(self, token: Token):
        self.is_token_allowed_condition(token).validate(token=token, state=self.state)
        self._transition_fn(token)

    def transition_sequence(self, tokens: list[Token]):
        for token in tokens:
            self.transition(token)

    def is_token_allowed_condition(self, token: AnyToken) -> TransitionCondition:
        # Bar or EndOfScore
        if isinstance(token, (Bar, EndOfScore)):
            max_position = max((v.position for v in self.state.voices), default=float("inf"))
            bar_condition = TransitionCondition.or_(
                ~self.state.voices_initialized_condition,  # only for first bar
                TransitionCondition.and_(
                    self.state.check_for_all_voices_condition(  # check that all voices can be finished
                        lambda v: TransitionCondition.and_(
                            ~v.in_structure_condition,  # no voice in chord or tuplet
                            TransitionCondition.or_(  # voices either not started or finished
                                TransitionCondition(lambda: v.position == 0, "pos = 0"),
                                TransitionCondition(
                                    lambda: v.position == max_position, "pos = max_dur"
                                ),
                            ),
                        )
                    ),
                ),
            )
            if isinstance(token, Bar):
                return bar_condition
            elif isinstance(token, EndOfScore):
                return TransitionCondition.and_(
                    bar_condition,  # Same conditions as for a Bar...
                    TransitionCondition(
                        lambda: len(self.state.tied_notes) == 0, "No notes tied"
                    ),  # but w/o ties
                )
            else:
                assert_never(token)

        # Note
        elif isinstance(token, Note):
            voice = self.state.voices[token.voice_idx]
            in_chord = voice.in_chord_condition
            is_grace = TransitionCondition(lambda: token.grace, "grace note")
            return TransitionCondition.and_(
                self.state.voices_initialized_condition,
                TransitionCondition.or_(
                    # In chord conditions
                    TransitionCondition.and_(
                        in_chord,
                        ~is_grace,
                        TransitionCondition(
                            lambda: token.duration == getattr(voice.chord, "duration", None),
                            "dur = chord_dur",
                            "note duration should match chord duration",
                        ),
                        *(
                            []
                            if self.allow_duplicate_notes_in_chord
                            else [
                                TransitionCondition(
                                    lambda: not token.note_tuple
                                    in getattr(voice.chord, "notes", set()),
                                    "note not already in chord",
                                )
                            ]
                        ),
                    ).with_alias("chord note conditions"),
                    # Not in chord conditions
                    TransitionCondition.and_(
                        ~in_chord,
                        TransitionCondition.or_(
                            # Grace note conditions
                            TransitionCondition.and_(
                                is_grace,
                                TransitionCondition(
                                    lambda: voice.position < voice.duration,
                                    "pos < dur",
                                    "space left",
                                ),
                            ),
                            # Not grace note conditions
                            TransitionCondition.and_(
                                ~is_grace,
                                TransitionCondition(
                                    lambda: token.duration.quarter_length
                                    <= voice.segment_remaining_duration,
                                    "dur <= remaining_dur",
                                ),
                            ),
                        ),
                    ).with_alias("non-chord note conditions"),
                ),
            )

        # Rest
        elif isinstance(token, Rest):
            voice = self.state.voices[token.voice_idx]
            measure_rest = TransitionCondition(lambda: token.is_measure, "measure rest")
            return TransitionCondition.and_(
                self.state.voices_initialized_condition,
                TransitionCondition.or_(
                    TransitionCondition.and_(
                        measure_rest,
                        (~voice.started_condition).with_alias("voice not started"),
                    ).with_alias("measure rest conditions"),
                    TransitionCondition.and_(
                        ~measure_rest,
                        ~voice.in_chord_condition,
                        TransitionCondition(
                            lambda: token.duration.quarter_length
                            <= voice.segment_remaining_duration,
                            "dur <= remaining_dur",
                            "enough space in segment",
                        ),
                    ).with_alias("regular rest conditions"),
                ),
            )

        # ChordStart
        elif isinstance(token, ChordStart):
            voice = self.state.voices[token.voice_idx]
            return TransitionCondition.and_(
                self.state.voices_initialized_condition,
                TransitionCondition.and_(
                    ~voice.in_chord_condition,
                    TransitionCondition(
                        lambda: token.duration.quarter_length <= voice.segment_remaining_duration,
                        "dur <= remaining_dur",
                    ),
                ),
            )

        # ChordEnd
        elif isinstance(token, ChordEnd):
            voice = self.state.voices[token.voice_idx]
            return TransitionCondition.and_(
                voice.in_chord_condition, getattr(voice.chord, "can_resolve_condition", None)
            )

        # TupletStart
        elif isinstance(token, TupletStart):
            voice = self.state.voices[token.voice_idx]
            return TransitionCondition.and_(
                self.state.voices_initialized_condition,
                TransitionCondition.and_(
                    ~voice.in_structure_condition,
                    TransitionCondition(
                        lambda: token.normal_duration
                        <= self.state.voices[token.voice_idx].segment_remaining_duration,
                        "norm_dur <= rem_dur",
                        "tuplet fits in remaining duration",
                    ),
                ),
            )

        # TupletEnd
        elif isinstance(token, TupletEnd):
            voice = self.state.voices[token.voice_idx]
            return TransitionCondition.and_(
                voice.in_tuplet_condition,
                TransitionCondition(
                    lambda: voice.segment_remaining_duration == 0, "rem_dur = 0", "tuplet finished"
                ),
            )

        # Errors
        else:
            raise ValueError(f"Unknown token: {token}")

    def _transition_fn(self, token: AnyToken):
        # Bar
        if isinstance(token, Bar):
            # Reset all voices to zero
            self.state.initialize_voices(
                token.duration
                if self.allow_longer_measures
                else min(token.duration, token.nominal_duration)
            )
            # Mark all ties as resolvable (after a bar line, all ties should be resolvable)
            for pitch, props in self.state.tied_notes.items():
                if props.min_resolution_pos != 0:  # Avoid resetting already resolvable ties
                    self.state.tied_notes[pitch] = TieProperties(props.note_tuple, Fraction(0))
            # Update the measure number
            if hasattr(token, "_partitura_elements"):
                self.state.measure_number = token._partitura_elements[0].number
            else:
                self.state.measure_number += 1

        # Note
        elif isinstance(token, Note):
            voice = self.state.voices[token.voice_idx]
            # If resolving a previous tie, remove the note from the tied notes
            if token.pitch in self.state.resolvable_tied_notes(voice.position):
                self.state.tied_notes.pop(token.pitch)
            # If tied, add the note to tied notes
            if token.tied:
                self.state.tied_notes[token.pitch] = TieProperties(
                    note_tuple=token.note_tuple,
                    min_resolution_pos=voice.position + token.duration.quarter_length,
                )
            # Only advance position if not in chord of grace note
            if voice.in_chord_condition:
                voice.chord.notes.append(token.note_tuple)
            elif token.grace:
                pass  # Do not advance position for grace notes
            else:
                voice.advance_position(token.duration.quarter_length)

        # Rest
        elif isinstance(token, Rest):
            voice = self.state.voices[token.voice_idx]
            if not token.is_measure:
                voice.advance_position(token.duration.quarter_length)
            else:
                voice.advance_position(voice.duration)

        # ChordStart
        elif isinstance(token, ChordStart):
            self.state.voices[token.voice_idx].chord = ChordState(token.duration)

        # ChordEnd
        elif isinstance(token, ChordEnd):
            voice = self.state.voices[token.voice_idx]
            voice.advance_position(voice.chord.duration.quarter_length)
            voice.chord = None

        # TupletStart
        elif isinstance(token, TupletStart):
            self.state.voices[token.voice_idx].tuplet = TupletCompletionState(
                n_actual=token.n_actual,
                n_normal=token.n_normal,
                unit_duration=token.unit_duration,
            )

        # TupletEnd
        elif isinstance(token, TupletEnd):
            self.state.voices[token.voice_idx].tuplet = None

        # EndOfScore
        elif isinstance(token, EndOfScore):
            pass

        # Errors
        else:
            raise ValueError(f"Unknown token: {token}")
