"""Explicit, versioned data vocabularies, separate from any model Word Head."""

from __future__ import annotations

import re

from cosmos_framework.data.generator.physword.common import ATOMIC_ACTIONS, digest

LEGACY_VOCABULARY_ID = "physword_atomic_v1"


def validate_vocabulary(vocabulary):
    if not isinstance(vocabulary, dict) or set(vocabulary) != {"id", "words", "definitions"}:
        raise ValueError("Vocabulary requires exactly id, words and definitions")
    if not isinstance(vocabulary["id"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", vocabulary["id"]):
        raise ValueError("Invalid vocabulary identity")
    words = vocabulary["words"]
    if (
        not isinstance(words, list)
        or not words
        or any(not isinstance(w, str) or not re.fullmatch(r"[a-z]+", w) for w in words)
    ):
        raise ValueError("Every primitive must be a single lowercase alphabetic word")
    if len(words) != len(set(words)) or {"idle", "unknown"}.intersection(words):
        raise ValueError("Vocabulary must be unique and exclude idle/unknown")
    definitions = vocabulary["definitions"]
    if (
        not isinstance(definitions, dict)
        or set(definitions) != set(words)
        or any(not isinstance(v, str) or not v.strip() for v in definitions.values())
    ):
        raise ValueError("Every vocabulary word needs exactly one nonempty definition")
    if vocabulary["id"] == LEGACY_VOCABULARY_ID and tuple(words) != ATOMIC_ACTIONS:
        raise ValueError("Cannot redefine the legacy sixteen-class vocabulary")
    return vocabulary


def vocabulary_for(annotation):
    if "vocabulary" in annotation:
        return validate_vocabulary(annotation["vocabulary"])
    return {"id": LEGACY_VOCABULARY_ID, "words": list(ATOMIC_ACTIONS), "definitions": {w: w for w in ATOMIC_ACTIONS}}


def vocabulary_hash(vocabulary):
    return digest({"id": vocabulary["id"], "words": vocabulary["words"]})


def require_legacy_model_vocabulary(sample):
    if (
        sample.get("pw_vocabulary_id", LEGACY_VOCABULARY_ID) != LEGACY_VOCABULARY_ID
        or tuple(sample.get("pw_vocabulary", ATOMIC_ACTIONS)) != ATOMIC_ACTIONS
    ):
        raise ValueError(
            "Scene vocabulary cannot feed the legacy 16-class M3 Word Head; explicitly adapt model vocabulary and per-arm targets first"
        )
