"""Module to count inaccuracies in ECLASS definitions."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openai import OpenAI
from spellchecker import SpellChecker

from preprocessing.csv_2_filter import STRUCTURAL_SET
from preprocessing.filter_non_semantic import (
    filter_definitions_missing, filter_definitions_missing_suffix)
from preprocessing.filter_spelling import (filter_chemical_compounds,
                                           filter_correct_spellings)
from utils.csv_io import load_csv
from utils.logger import LoggerFactory
from utils.lookup import lookup_definition

MISSING_SET = frozenset(filter_definitions_missing)
MISSING_SUFFIXES = tuple(filter_definitions_missing_suffix)
STRUCTURAL_SET = frozenset(STRUCTURAL_SET)
SPELLING_SET = frozenset(filter_correct_spellings)
CHEMICAL_SET = frozenset(filter_chemical_compounds)


@dataclass(frozen=True)
class SegmentStats:
    """Per-segment ECLASS inaccuracy stats."""

    count: int
    definitions: int
    inaccurate_ids: List[str]


@dataclass(frozen=True)
class CheckerStats:
    """Aggregated ECLASS inaccuracy results across several segments for one checker."""

    name: str
    data_dir: str
    by_segment: Dict[int, SegmentStats]


class InaccuracyChecker(ABC):
    """Abstract interface for returning ECLASS definition inaccuracies."""

    def __init__(self, name: str, data_dir: str):
        self.name = name
        self.data_dir = data_dir

    @abstractmethod
    def find_inaccuracies(self, ids, names, definitions) -> List[str]:
        """Return a list of IRDIs for all definitions that contain this inaccuracy type"""

        raise NotImplementedError

    def run(
            self,
            base_dir: str,
            segments: List[int],
            exceptions: List[int],
            logger: logging.Logger,
    ) -> CheckerStats:
        """Execute this checker once per segment and once on the combined file and return the aggregated ECLASS
        inaccuracy result."""

        format_dir = Path(base_dir) / self.data_dir
        exc = set(exceptions)
        by_segment: Dict[int, SegmentStats] = {}
        ALL_SEGMENT = 0  # Convention, key "0" represents the stats for the combined file "eclass-0.csv"

        # Per segment and combined run
        for seg in segments + [ALL_SEGMENT]:
            if seg in exc:
                logger.warning(f"Skipping segment {seg}.")
                continue

            # Load data
            segment_path = format_dir / f"eclass-{seg}.csv"
            df = load_csv(segment_path, "id", "preferred-name", "definition", logger)
            if df is None:
                continue
            df_iter = (
                df[["id", "preferred-name", "definition"]]
                .dropna(subset=["id", "preferred-name", "definition"])
                .astype({"id": str, "preferred-name": str, "definition": str})
                .itertuples(index=False, name=None)
            )
            triplets = list(df_iter)
            if not triplets:
                by_segment[seg] = SegmentStats(count=0, definitions=0, inaccurate_ids=[])
                continue

            # Find inaccuracies
            ids, names, definitions = (list(t) for t in zip(*triplets))
            inaccurate_ids = self.find_inaccuracies(ids, names, definitions)
            seg_inacc_count = len(inaccurate_ids)
            seg_def_count = len(definitions)

            by_segment[seg] = SegmentStats(
                count=seg_inacc_count,
                definitions=seg_def_count,
                inaccurate_ids=inaccurate_ids
            )

            # Log results
            pct = seg_inacc_count / seg_def_count * 100 if seg_def_count > 0 else 0.0
            seg_label = "combined" if seg == 0 else str(seg)
            logger.info(
                "[%s] Segment %s (%s): total = %d, inaccuracies = %d (%.2f%%)",
                self.name,
                seg_label,
                self.data_dir,
                seg_def_count,
                seg_inacc_count,
                pct
            )

        # Return aggregated results
        return CheckerStats(
            name=self.name,
            data_dir=self.data_dir,
            by_segment=by_segment
        )


class MissingDefinitionChecker(InaccuracyChecker):
    """Count ECLASS definitions considered missing via exact or suffix filter rules."""

    def __init__(self, data_dir: str):
        super().__init__(name="missing-definition", data_dir=data_dir)

    def find_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> List[str]:
        return [
            id for id, definition in zip(ids, definitions)
            if not definition or definition in MISSING_SET or definition.endswith(MISSING_SUFFIXES)
        ]


class StructuralDefinitionChecker(InaccuracyChecker):
    """Count ECLASS definitions considered structural via exact filter rules."""

    def __init__(self, data_dir: str):
        super().__init__(name="structural-definition", data_dir=data_dir)

    def find_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> List[str]:
        return [
            id for id, d in zip(ids, definitions)
            if d in STRUCTURAL_SET
        ]


class NoFullStopChecker(InaccuracyChecker):
    """Count ECLASS definitions that do not end with a full stop."""

    def __init__(self, data_dir: str):
        super().__init__(name="no-full-stop", data_dir=data_dir)

    def find_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> List[str]:
        return [
            id for id, d in zip(ids, definitions)
            if d and not d.endswith(".")
        ]


class NoCapitalStartChecker(InaccuracyChecker):
    """Count definitions where the first alphabetic character is lowercase."""

    def __init__(self, data_dir: str):
        super().__init__(name="no-capital-start", data_dir=data_dir)

    @staticmethod
    def _first_alpha_char(s: str) -> Optional[str]:
        for character in s:
            if character.isalpha():
                return character
        return None

    def find_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> List[str]:
        return [
            id for id, d in zip(ids, definitions)
            if (
                    (ch := self._first_alpha_char(d)) is not None
                    and ch.islower()
            )
        ]


class SpellingChecker(InaccuracyChecker):
    """Count definitions that contain at least one likely spelling mistake."""

    TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß']+[A-Za-zÄÖÜäöüß'\-]*")

    def __init__(
            self,
            data_dir: str,
            languages: Iterable[str] = ("en",),
            ignore_all_caps: bool = True,
            ignore_with_digits: bool = True,
            ignore_chemical_compounds: bool = True,
            min_len: int = 4
    ):
        super().__init__(name="spelling", data_dir=data_dir)
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

    def find_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> List[str]:
        inaccurate_ids: List[str] = []

        for id, definition in zip(ids, definitions):
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
                    inaccurate_ids.append(id)
                    break

        return inaccurate_ids


class DuplicateDefinitionChecker(InaccuracyChecker):
    """Count definitions that occur more than once."""

    def __init__(self, data_dir: str):
        super().__init__(name="duplicate-definition", data_dir=data_dir)

    def find_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> List[str]:
        norm_to_ids = {}
        for id, definition in zip(ids, definitions):
            norm_to_ids.setdefault(definition, []).append(id)

        # Count every occurrence in groups with size > 1
        duplicate_ids = []
        for id_list in norm_to_ids.values():
            if len(id_list) > 1:
                duplicate_ids.extend(id_list)
        return duplicate_ids


class DuplicateNameDefinitionChecker(InaccuracyChecker):
    """Count definitions that occur more than once together with the same preferred-name."""

    def __init__(self, data_dir: str):
        super().__init__(name="duplicate-name-definition", data_dir=data_dir)

    def find_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> List[str]:
        pair_to_ids: Dict[tuple, List[str]] = {}
        for id_, name, definition in zip(ids, names, definitions):
            key = (name, definition)
            pair_to_ids.setdefault(key, []).append(id_)

        # Count every occurrence in groups with size > 1
        duplicate_ids: List[str] = []
        for id_list in pair_to_ids.values():
            if len(id_list) > 1:
                duplicate_ids.extend(id_list)

        return duplicate_ids


class NameEqualsDefinitionChecker(InaccuracyChecker):
    """Find definitions where the name and the definition are identical."""

    def __init__(self, data_dir: str):
        super().__init__(name="name-equals-definition", data_dir=data_dir)

    def find_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> List[str]:
        invalid_ids = []
        for id_, name, definition in zip(ids, names, definitions):
            norm_name = re.sub(r"\s+", " ", name.lower().strip())
            norm_def = re.sub(r"\s+", " ", definition.lower().strip())
            if norm_name == norm_def:
                invalid_ids.append(id_)

        return invalid_ids


class HiddenCharWatermarkChecker(InaccuracyChecker):
    """Count definitions containing hidden Unicode characters often used as watermarks by LLMs."""

    def __init__(self, data_dir: str):
        super().__init__(name="hidden-watermark", data_dir=data_dir)
        self.HIDDEN_CHARS = {
            "\u00A0": "NO-BREAK SPACE",
            "\u200B": "ZERO WIDTH SPACE",
            "\u200C": "ZERO WIDTH NON-JOINER",
            "\u200D": "ZERO WIDTH JOINER",
            "\u202F": "NARROW NO-BREAK SPACE",
            "\u2060": "WORD JOINER",
            "\uFEFF": "ZERO WIDTH NO-BREAK SPACE",
        }

    def _contains_hidden_char(self, s: str) -> bool:
        return any(character in s for character in self.HIDDEN_CHARS)

    def find_inaccuracies(self, ids: List[str], names: List[str], definitions: List[str]) -> List[str]:
        inaccurate_ids = []
        for id, definition in zip(ids, definitions):
            if definition and self._contains_hidden_char(definition):
                inaccurate_ids.append(id)
        return inaccurate_ids


def run_checkers(
        checkers: List[InaccuracyChecker],
        base_dir: str,
        segments: List[int],
        exceptions: List[int],
        logger: logging.Logger,
) -> Dict[str, CheckerStats]:
    """Run multiple inaccuracy checkers and return their aggregated results."""

    results: Dict[str, CheckerStats] = {}
    for checker in checkers:
        stats = checker.run(
            base_dir=base_dir,
            segments=segments,
            exceptions=exceptions,
            logger=logger,
        )
        results[checker.name] = stats
    return results


if __name__ == "__main__":
    mode = "classes"  # Categorisation classes via "classes", properties via "properties"

    # Initialise
    logger = LoggerFactory.get_logger(__name__)
    file_handler = logging.FileHandler(f"eclass-{mode}-inaccuracy-run.txt", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Initialising ECLASS inaccuracy checkers ...")
    base_dir = f"../../data/extracted-{mode}"
    segments = list(range(13, 52)) + [90]
    exceptions = []

    # Run all checkers
    checkers = [
        MissingDefinitionChecker(data_dir=f"1-original-{mode}"),
        StructuralDefinitionChecker(data_dir=f"1-original-{mode}"),
        NoFullStopChecker(data_dir=f"2-filtered-{mode}"),
        NoCapitalStartChecker(data_dir=f"2-filtered-{mode}"),
        #SpellingChecker(data_dir=f"2-filtered-{mode}"),
        DuplicateDefinitionChecker(data_dir=f"3-normalised-{mode}"),
        DuplicateNameDefinitionChecker(data_dir=f"3-normalised-{mode}"),
        NameEqualsDefinitionChecker(data_dir=f"3-normalised-{mode}"),
        HiddenCharWatermarkChecker(data_dir=f"2-filtered-{mode}")
    ]
    results = run_checkers(
        checkers=checkers,
        base_dir=base_dir,
        segments=segments,
        exceptions=exceptions,
        logger=logger
    )

    # Log results
    logger.info("=== Aggregated Results ===")
    for checker_name, stats in results.items():
        overall = stats.by_segment.get(0)
        pct = (overall.count / overall.definitions * 100) if overall.definitions else 0.0
        logger.info(
            "[%s | %s] Combined -> definitions: %d | inaccuracies: %d (%.2f%%)",
            stats.name,
            stats.data_dir,
            overall.definitions,
            overall.count,
            pct
        )

    # Write watermark and spelling files
    watermark_checker = results.get("hidden-watermark")
    if watermark_checker:
        overall = watermark_checker.by_segment.get(0)
        if overall and overall.inaccurate_ids:
            output_dir = base_dir + "/" + watermark_checker.data_dir
            with open(f"eclass-{mode}-definitions-watermarks.txt", "w", encoding="utf-8") as f:
                for irdi in overall.inaccurate_ids:
                    definition = lookup_definition(irdi, output_dir)
                    f.write(f"{irdi}: {definition}\n")

    spelling_checker = results.get("spelling")
    if spelling_checker:
        overall = spelling_checker.by_segment.get(0)
        if overall and overall.inaccurate_ids:
            output_dir = base_dir + "/" + spelling_checker.data_dir
            with open(f"eclass-{mode}-definitions-misspelled.txt", "w", encoding="utf-8") as f:
                for irdi in overall.inaccurate_ids:
                    definition = lookup_definition(irdi, output_dir)
                    f.write(f"{irdi}: {definition}\n")
