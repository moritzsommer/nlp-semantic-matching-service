"""Module to count inaccuracies in ECLASS definitions."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from spellchecker import SpellChecker
from src.embedding.filter import (filter_definitions_missing,
                                  filter_definitions_missing_suffix,
                                  filter_definitions_structural)
from utils.logger import LoggerFactory

from evaluation.filter import (filter_chemical_compounds,
                               filter_correct_spellings)

MISSING_SET = frozenset(filter_definitions_missing)
MISSING_SUFFIXES = tuple(filter_definitions_missing_suffix)
STRUCTURAL_SET = frozenset(filter_definitions_structural)
SPELLING_SET = frozenset(filter_correct_spellings)
CHEMICAL_SET = frozenset(filter_chemical_compounds)


@dataclass(frozen=True)
class SegmentStats:
    """Per-segment ECLASS inaccuracy stats."""

    count: int
    definitions: int


@dataclass(frozen=True)
class CheckerStats:
    """Aggregated ECLASS inaccuracy results across several segments for one checker."""

    name: str
    semantic_definitions_only: bool
    total_inaccuracies: int
    total_definitions: int
    by_segment: Dict[int, SegmentStats]


def _load_segment(input_path: str, seg: int, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Load an ECLASS segment and validate required columns."""

    try:
        database = pd.read_csv(input_path.format(segment=seg), sep=",")
        #logger.info(f"Database loaded from {input_path} with {len(database)} rows.")
    except Exception as e:
        logger.error(f"Failed to read file: {input_path}, Error: {e}")
        return None

    required = {"id", "preferred_name", "definition"}
    missing = required - set(database.columns)
    if missing:
        logger.error(f"Missing required column: {missing}")
        return None

    return database


class InaccuracyChecker(ABC):
    """Abstract interface for counting ECLASS definition inaccuracies."""

    def __init__(self, name: str, semantic_definitions_only: bool = True, normalise_strings: bool = True):
        self.name = name
        self.semantic_definitions_only = semantic_definitions_only
        self.normalise_strings = normalise_strings

    @abstractmethod
    def count_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> int:
        """Return count for this ECLASS definition inaccuracy inside one segment."""

        raise NotImplementedError

    @staticmethod
    def _trim_wrappers(s: str) -> str:
        s = s.strip()
        leading = '([{"\'«“‚‹'
        trailing = ')]}"\'»”’›'
        while s and s[0] in leading:
            s = s[1:].lstrip()
        while s and s[-1] in trailing:
            s = s[:-1].rstrip()
        return s

    @staticmethod
    def _is_semantic(definition: str) -> bool:
        if not definition:
            return False
        if definition in MISSING_SET:
            return False
        if definition.endswith(MISSING_SUFFIXES):
            return False
        if definition in STRUCTURAL_SET:
            return False
        return True

    @staticmethod
    def filter_semantic_pairs(pairs: Iterable[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Filter definitions, their names and ids by excluding entries with missing and placeholder definitions."""

        out = []
        for id, name, definition in pairs:
            if InaccuracyChecker._is_semantic(definition):
                out.append((id, name, definition))
        return out

    @staticmethod
    def normalise(s: Optional[str]) -> str:
        """Return a normalised string."""

        # Normalise by mapping none to an empty string
        if not s:
            return ""

        # Normalise by turning space-like characters into normal spaces
        trans = {
            0x00A0: 0x0020,  # NO-BREAK SPACE
            0x202F: 0x0020,  # NARROW NO-BREAK SPACE
        }
        s = s.translate(trans)

        # Normalise by stripping the string
        s = s.strip()

        # Normalise by removing wrapping characters
        s = InaccuracyChecker._trim_wrappers(s)

        # Normalise by collapsing whitespaces
        s = " ".join(s.split())

        # Normalise by using lower case only
        s = s.lower()

        return s

    def run(
            self,
            input_path: str,
            segments: List[int],
            exceptions: List[int],
            logger: logging.Logger,
    ) -> CheckerStats:
        """Execute this checker across several segments and return the aggregated ECLASS inaccuracy result."""

        exc = set(exceptions)
        by_segment: Dict[int, SegmentStats] = {}
        total_inaccuracy_count = 0
        total_definition_count = 0

        for seg in segments:
            if seg in exc:
                logger.warning(f"Skipping segment {seg}.")
                continue

            # Load data
            database_segment = _load_segment(input_path, seg, logger)
            if database_segment is None:
                continue

            database_segment = (
                database_segment[["id", "preferred_name", "definition"]].
                dropna(subset=["id", "preferred_name", "definition"]).
                astype({"id": str, "preferred_name": str, "definition": str}).
                itertuples(index=False, name=None)
            )

            # Remove non-semantic definitions
            if self.semantic_definitions_only:
                database_segment = self.filter_semantic_pairs(database_segment)

            # Normalise all strings
            if self.normalise_strings:
                normed = []
                for id, name, definition in database_segment:
                    name_n = self.normalise(name)
                    definition_n = self.normalise(definition)
                    normed.append((id, name_n, definition_n))
                database_segment = normed

            # Count inaccuracies
            ids, names, definitions = (list(t) for t in zip(*database_segment))
            segment_definition_count = len(definitions)
            segment_inaccuracy_count = self.count_inaccuracies(ids, names, definitions)

            by_segment[seg] = SegmentStats(count=segment_inaccuracy_count, definitions=segment_definition_count)
            total_inaccuracy_count += segment_inaccuracy_count
            total_definition_count += segment_definition_count

            # Log the output
            filtered_flag = "yes" if self.semantic_definitions_only else "no"
            pct = segment_inaccuracy_count / segment_definition_count * 100 if segment_definition_count > 0 else 0
            logger.info(
                "[%s] Segment %s: filtered = %s, total = %d, inaccuracies = %d (%.2f%%)",
                self.name, seg, filtered_flag, segment_definition_count, segment_inaccuracy_count, pct
            )

        return CheckerStats(
            name=self.name,
            semantic_definitions_only=self.semantic_definitions_only,
            total_inaccuracies=total_inaccuracy_count,
            total_definitions=total_definition_count,
            by_segment=by_segment
        )


class MissingDefinitionChecker(InaccuracyChecker):
    """Count ECLASS definitions considered missing via exact or suffix filter rules. Consider all definitions."""

    def __init__(self):
        super().__init__(name="missing-definition", semantic_definitions_only=False, normalise_strings=False)

    def count_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> int:
        return sum(1 for d in definitions if d in MISSING_SET or d.endswith(MISSING_SUFFIXES))


class StructuralDefinitionChecker(InaccuracyChecker):
    """Count ECLASS definitions considered structural via exact filter rules. Consider all definitions."""

    def __init__(self):
        super().__init__(name="structural-definition", semantic_definitions_only=False, normalise_strings=False)

    def count_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> int:
        return sum(1 for d in definitions if d in STRUCTURAL_SET)


class NoFullStopChecker(InaccuracyChecker):
    """Count ECLASS definitions that do not end with a full stop. Consider only semantic definitions."""

    def __init__(self):
        super().__init__(name="no-full-stop")

    def count_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> int:
        count = 0
        for definition in definitions:
            if definition and not definition.endswith("."):
                count += 1
        return count


class NoCapitalStartChecker(InaccuracyChecker):
    """Count definitions where the first alphabetic character is lowercase."""

    def __init__(self):
        super().__init__(name="no-capital-start", normalise_strings=False)

    @staticmethod
    def _first_alpha_char(s: str) -> Optional[str]:
        for character in s:
            if character.isalpha():
                return character
        return None

    def count_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> int:
        count = 0
        for definition in definitions:
            character = self._first_alpha_char(definition)
            if character is not None and character.islower():
                count += 1
        return count


class SpellingChecker(InaccuracyChecker):
    """Count definitions that contain at least one likely spelling mistake."""

    TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß']+[A-Za-zÄÖÜäöüß'\-]*")

    def __init__(
            self,
            languages: Iterable[str] = ("en",),
            ignore_all_caps: bool = True,
            ignore_with_digits: bool = True,
            ignore_chemical_compounds: bool = True,
            min_len: int = 4
    ):
        super().__init__(name="spelling")
        self.whitelist = SPELLING_SET
        self.llm_cache: Dict[str, Optional[bool]] = {}
        self.client = OpenAI(api_key="PLACEHOLDER")  # OpenAI API key needed

        self.ignore_all_caps = ignore_all_caps
        self.ignore_with_digits = ignore_with_digits
        self.ignore_chemical_compounds = ignore_chemical_compounds
        self.min_len = min_len

        # Build one SpellChecker per language
        self.spellers: List[SpellChecker] = []
        for lang in languages:
            sp = SpellChecker(language=lang)
            if self.whitelist:
                sp.word_frequency.load_words(self.whitelist)
            self.spellers.append(sp)

        open("eclass-definitions-misspelled.txt", "w", encoding="utf-8").close()

    @staticmethod
    def _check_chemical_token(
            token: str,
            min_len: int = 4
    ) -> bool:
        # Check for typical chemical expressions
        if len(token) < min_len:
            return False
        if any(m in token for m in CHEMICAL_SET):
            return True

        # Check for Locants, primes and greek letters
        _CHEM_LOCANT_RE = re.compile(
            r"""(?xi)
            ^[NOPS]- |  # N- / O- / P- / S- locants
            \d+(?:,\d+|'|′|″)*- |  # 4- / 4,4'- / primes
            \b(?:alpha|beta|gamma|delta)\b  # Greek locants
            """
        )
        if _CHEM_LOCANT_RE.search(token):
            return True

        # Check for parentheses with hyphens inside
        _CHEM_PAREN_HYPHEN_RE = re.compile(r".*[()]\S*-\S*[()].*")
        if _CHEM_PAREN_HYPHEN_RE.match(token):
            return True

        # Check for letters and digits with hyphens
        if re.search(r"[A-Za-z]\d|\d[A-Za-z]", token) and "-" in token:
            return True
        return False

    def _tokenize(self, text: str) -> List[str]:
        return self.TOKEN_RE.findall(text)

    def _is_ignored(self, token: str) -> bool:
        if len(token) < self.min_len:
            return True
        if "-" in token:
            return True
        if "'" in token:
            return True
        if self.ignore_all_caps and token.isupper():
            return True
        if self.ignore_with_digits and any(ch.isdigit() for ch in token):
            return True
        if self.ignore_chemical_compounds and self._check_chemical_token(token):
            return True
        return False

    def _is_misspelled(self, token: str) -> bool:
        if token in self.whitelist:
            return False
        for sp in self.spellers:  # If any speller knows the word, it is spelled correctly
            if not sp.unknown([token]):
                return False
        return True

    def _confirm_with_llm(self, token: str, sentence: str) -> Optional[bool]:
        system_msg = (
            "You are a strict English spelling judge. Consider ONLY standard American and British English (US/UK). "
            "Accept both US/UK variants (e.g., color/colour, center/centre). Accept proper nouns, technical terms,"
            "chemical names, alphanumeric model codes and common hyphenation variants as correct. Reply with a single"
            "word: true or false."
        )
        user_msg = (
            f"TOKEN: {token}\n"
            f"SENTENCE: {sentence}\n"
            f"Is TOKEN misspelled in this sentence? Reply only: true or false."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=2,
        )

        content = (resp.choices[0].message.content or "").strip().lower()
        if content in {"true", "false"}:
            return content == "true"
        return None

    def count_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> int:
        defs_with_errors = 0
        for id, definition in zip(ids, definitions):
            has_error = False
            for token in self._tokenize(definition):
                # Check if the current word should be skipped
                if self._is_ignored(token):
                    continue

                # Apply local spelling heuristic to the current word
                if not self._is_misspelled(token):
                    continue

                # If the local heuristic finds a spelling mistake in the current word, apply LLM as verdict tie-breaker
                key = token.lower()
                verdict = self.llm_cache.get(key)  # Look up verdict in cache or call LLM
                if verdict is None:
                    verdict = self._confirm_with_llm(token, definition)
                    self.llm_cache[key] = verdict
                if verdict is False:
                    continue
                else:
                    has_error = True
                    with open("eclass-definitions-misspelled.txt", "a", encoding="utf-8") as f:
                        f.write(f"{id} - {key}: {definition}\n")
                    break

            if has_error:
                defs_with_errors += 1

        return defs_with_errors


class DuplicateDefinitionChecker(InaccuracyChecker):
    """Count definitions that occur more than once in the same segment."""

    def __init__(self):
        super().__init__(name="duplicate-definition")

    def count_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> int:
        norm_to_ids = {}
        for id, definition in zip(ids, definitions):
            norm_to_ids.setdefault(definition, []).append(id)

        # Count every occurrence in groups with size > 1
        dup_count = sum(len(value) for value in norm_to_ids.values() if len(value) > 1)
        return dup_count


class HiddenCharWatermarkChecker(InaccuracyChecker):
    """Count definitions containing hidden Unicode characters often used as watermarks by LLMs."""

    def __init__(self):
        super().__init__(name="hidden-watermark", normalise_strings=False)
        self.HIDDEN_CHARS = {
            "\u00A0": "NO-BREAK SPACE",
            "\u200B": "ZERO WIDTH SPACE",
            "\u200C": "ZERO WIDTH NON-JOINER",
            "\u200D": "ZERO WIDTH JOINER",
            "\u202F": "NARROW NO-BREAK SPACE",
            "\u2060": "WORD JOINER",
            "\uFEFF": "ZERO WIDTH NO-BREAK SPACE",
        }

        open("eclass-definitions-watermarks.txt", "w", encoding="utf-8").close()

    def _contains_hidden_char(self, s: str) -> bool:
        return any(character in s for character in self.HIDDEN_CHARS)

    def count_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> int:
        count = 0
        for id, definition in zip(ids, definitions):
            if definition and self._contains_hidden_char(definition):
                count += 1
                with open("eclass-definitions-watermarks.txt", "a", encoding="utf-8") as f:
                    f.write(f"{id}: {definition}\n")
        return count


def run_checkers(
        checkers: List[InaccuracyChecker],
        input_path: str,
        segments: List[int],
        exceptions: List[int],
        logger: logging.Logger,
) -> Dict[str, CheckerStats]:
    """Run multiple checkers and return results keyed by checker name."""

    results: Dict[str, CheckerStats] = {}
    for checker in checkers:
        res = checker.run(input_path, segments, exceptions, logger)
        results[res.name] = res
    return results


if __name__ == "__main__":
    logger = LoggerFactory.get_logger(__name__)
    logger.info("Initialising ECLASS inaccuracy run ...")

    segments = list(range(13, 52)) + [90]
    exceptions = []
    input_path = "../../data/extracted/properties/eclass-{segment}.csv"

    # Run all checkers
    checkers = [
        MissingDefinitionChecker(),
        StructuralDefinitionChecker(),
        NoFullStopChecker(),
        NoCapitalStartChecker(),
        #SpellingChecker(),
        DuplicateDefinitionChecker(),
        HiddenCharWatermarkChecker(),
    ]
    results = run_checkers(checkers, input_path, segments, exceptions, logger)

    # Log results
    for name, res in results.items():
        filtered_flag = "yes" if res.semantic_definitions_only else "no"
        pct = (res.total_inaccuracies / res.total_definitions * 100.0) if res.total_definitions else 0.0
        logger.info(
            "[%s] Overall: filtered = %s, total = %d, inaccuracies = %d (%.2f%%)",
            name, filtered_flag, res.total_definitions, res.total_inaccuracies, pct
        )
