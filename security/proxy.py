"""Air-gapped anonymization proxy.

Scrubs company names, executive names, currency figures, contract IDs and
common PII using spaCy NER plus regex rules and a learned deny-list, so every
occurrence of an entity is replaced consistently. Placeholders are
deterministic HMAC tokens (stable across runs when ``APEX_PROXY_KEY`` is
set). The re-identification vault lives only in process memory.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Pattern, Tuple

import spacy

from security.network_guard import local_processing_guarantee


class LeakageError(RuntimeError):
    """Raised when sensitive material survives scrubbing at the egress gate."""


@dataclass(frozen=True)
class Span:
    """A detected sensitive span. Priority: 0=deny-list, 1=regex, 2=NER."""

    start: int
    end: int
    label: str
    priority: int
    text: str


_PREFIXES = "MSA|SOW|NDA|CTR|AGR|SLA|LIC|DPA|EULA|PO"
_CONTRACT_PREFIXED = re.compile(
    rf"\b(?:{_PREFIXES})[-/ ]?(?:[A-Z]{{1,4}}[-/])?(?:\d{{2,4}}[-/])?\d{{3,8}}\b"
)
_CONTRACT_LABELLED = re.compile(
    r"\b(?:contract|agreement|purchase\s+order|order)\s+(?:no\.?|number|#)\s*"
    r"((?-i:(?=[A-Z0-9\-/]*\d)[A-Z0-9][A-Z0-9\-/]{3,}))",
    re.I,
)
_CURRENCY = re.compile(
    r"(?:(?:US\$|USD|EUR|GBP|INR|CAD|AUD|CHF|JPY|[$€£₹¥])\s?\d[\d,]*(?:\.\d+)?"
    r"(?:\s?(?:thousand|million|billion|trillion|mm|bn|k|m|b)\b)?)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s(?:thousand|million|billion|trillion)\s"
    r"(?:US\s|U\.S\.\s)?(?:dollars|euros|pounds|rupees)\b)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|INR)\b)",
    re.I,
)

# (label, pattern, capture-group index)
_RULES: Tuple[Tuple[str, Pattern[str], int], ...] = (
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), 0),
    ("URL", re.compile(r"https?://[^\s)>\]]+"), 0),
    ("CONTRACT_ID", _CONTRACT_PREFIXED, 0),
    ("CONTRACT_ID", _CONTRACT_LABELLED, 1),
    ("TAX_ID", re.compile(r"\b\d{2}-\d{7}\b"), 0),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.\-])?(?:\(\d{3}\)\s?|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}(?!\d)"), 0),
    ("CURRENCY", _CURRENCY, 0),
)

_PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+_[0-9A-F]{6}\]")
_CORP_SUFFIX = re.compile(
    r"[,\s]+(?:inc|incorporated|corp|corporation|llc|ltd|limited|plc|co|company|gmbh|ag|sa|holdings|group)\.?$",
    re.I,
)


def _canon_key(label: str, text: str) -> str:
    """Normalize a surface form so spelling variants share one placeholder."""
    t = re.sub(r"\s+", " ", text).strip(" \t\r\n,.;:()\"'“”").casefold()
    if label == "COMPANY":
        t = re.sub(r"^the\s+", "", t)
        previous = None
        while previous != t:
            previous = t
            stripped = _CORP_SUFFIX.sub("", t)
            t = stripped or t
    elif label == "CURRENCY":
        t = t.replace(" ", "")
    return t


class AirGappedProxy:
    """Local-only PII/entity scrubber with an egress leakage gate."""

    LABEL_MAP = {"ORG": "COMPANY", "PERSON": "EXEC", "MONEY": "CURRENCY"}
    DEFAULT_IGNORE = frozenset(
        {
            "sec", "gaap", "ifrs", "fasb", "irs", "nyse", "nasdaq", "the company", "company",
            "board", "board of directors", "audit committee", "congress", "ceo", "cfo", "cto",
            "coo", "ebitda", "u.s.", "us", "ftc", "doj", "gdpr", "hsr", "hart-scott-rodino",
            "securities and exchange commission", "the securities and exchange commission",
            "financial accounting standards board", "federal trade commission",
            "department of justice",
        }
    )

    def __init__(
        self,
        spacy_model: str = "en_core_web_lg",
        secret_key: Optional[bytes] = None,
        known_entities: Optional[Mapping[str, Iterable[str]]] = None,
        ignore: Optional[Iterable[str]] = None,
        nlp: Optional[Any] = None,
    ) -> None:
        """Create the proxy.

        Args:
            spacy_model: Locally installed spaCy pipeline name.
            secret_key: HMAC key for placeholders. Falls back to the
                ``APEX_PROXY_KEY`` env var, then to an ephemeral random key.
            known_entities: ``{"COMPANY": [...], "EXEC": [...]}`` seed names
                guaranteed to be scrubbed even when NER misses them.
            ignore: Extra surface forms (case-insensitive) never to scrub.
            nlp: Optional pre-loaded spaCy pipeline (for reuse/testing).
        """
        env_key = os.environ.get("APEX_PROXY_KEY", "").encode()
        self._key: bytes = secret_key or env_key or secrets.token_bytes(32)
        self._nlp = nlp or self._load_model(spacy_model)
        self._ignore = frozenset(self.DEFAULT_IGNORE | {i.casefold() for i in (ignore or ())})
        self._known: Dict[str, str] = {}
        self._alias: Dict[str, str] = {}
        self._vault: Dict[str, str] = {}
        self._deny_re: Optional[Pattern[str]] = None
        self._counts: Counter = Counter()
        self._lock = threading.RLock()
        for label, names in (known_entities or {}).items():
            self.register(label, names)

    @staticmethod
    def _load_model(name: str) -> Any:
        with local_processing_guarantee(allow_loopback=False):
            try:
                return spacy.load(name, disable=["lemmatizer"])
            except OSError as exc:
                raise RuntimeError(
                    f"spaCy model '{name}' is not installed locally. Pre-install on a staging "
                    f"host with: python -m spacy download {name}"
                ) from exc

    # ------------------------------------------------------------------ registry
    def register(self, label: str, names: Iterable[str]) -> None:
        """Add names to the deny-list so all their occurrences are scrubbed."""
        with self._lock:
            for raw in names:
                name = re.sub(r"\s+", " ", raw.strip())
                if len(name) < 2:
                    continue
                self._known[name.casefold()] = label
                canon = _canon_key(label, name)
                if canon and canon != name.casefold():
                    self._known.setdefault(canon, label)
                parts = [p.strip(".,") for p in name.split()]
                if label == "EXEC" and len(parts) >= 2:
                    for part in (parts[0], parts[-1]):
                        if len(part) >= 4:
                            self._known.setdefault(part.casefold(), label)
                            self._alias[_canon_key(label, part)] = canon
            self._deny_re = None

    def _deny_pattern(self) -> Optional[Pattern[str]]:
        with self._lock:
            if self._deny_re is None and self._known:
                keys = sorted(self._known, key=len, reverse=True)
                body = "|".join(re.escape(k).replace(r"\ ", r"\s+") for k in keys)
                self._deny_re = re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.I)
            return self._deny_re

    # ----------------------------------------------------------------- detection
    def detect(self, text: str) -> List[Span]:
        """Return non-overlapping sensitive spans found in ``text``."""
        spans: List[Span] = []
        for label, pattern, group in _RULES:
            for match in pattern.finditer(text):
                s, e = match.span(group)
                if s >= 0 and e > s:
                    spans.append(Span(s, e, label, 1, text[s:e]))

        if text.strip():
            if len(text) >= self._nlp.max_length:
                self._nlp.max_length = len(text) + 1000
            for ent in self._nlp(text).ents:
                label = self.LABEL_MAP.get(ent.label_)
                if label is None:
                    continue
                s, e = ent.start_char, ent.end_char
                surface = text[s:e]
                if surface.endswith(("'s", "’s")):
                    e -= 2
                    surface = text[s:e]
                surface = surface.strip()
                if len(surface) < 3 or surface.casefold() in self._ignore or surface.isdigit():
                    continue
                if not (surface[0].isupper() or surface[0].isdigit() or surface[0] in "$€£₹¥"):
                    continue
                spans.append(Span(s, e, label, 2, text[s:e]))
                if label != "CURRENCY":
                    self.register(label, [surface])

        deny = self._deny_pattern()
        if deny is not None:
            for match in deny.finditer(text):
                key = re.sub(r"\s+", " ", match.group(0)).casefold()
                label = self._known.get(key)
                if label and key not in self._ignore:
                    spans.append(Span(match.start(), match.end(), label, 0, match.group(0)))
        return self._resolve(spans)

    @staticmethod
    def _resolve(spans: List[Span]) -> List[Span]:
        accepted: List[Span] = []
        for span in sorted(spans, key=lambda s: (s.priority, -(s.end - s.start), s.start)):
            if all(span.end <= a.start or span.start >= a.end for a in accepted):
                accepted.append(span)
        return sorted(accepted, key=lambda s: s.start)

    # ------------------------------------------------------------- substitution
    def placeholder_for(self, label: str, original: str) -> str:
        """Return the deterministic placeholder for an entity and vault the original."""
        key = _canon_key(label, original)
        key = self._alias.get(key, key)
        digest = hmac.new(self._key, f"{label}|{key}".encode(), hashlib.sha256).hexdigest()[:6].upper()
        token = f"[{label}_{digest}]"
        with self._lock:
            existing = self._vault.get(token)
            if existing is None or len(original.strip()) > len(existing):
                self._vault[token] = original.strip()
        return token

    def scrub(self, text: str) -> Tuple[str, List[Span]]:
        """Anonymize ``text`` and also return the spans that were replaced."""
        if not text:
            return text, []
        spans = self.detect(text)
        out: List[str] = []
        cursor = 0
        for span in spans:
            out.append(text[cursor:span.start])
            out.append(self.placeholder_for(span.label, span.text))
            cursor = span.end
        out.append(text[cursor:])
        self._counts.update(span.label for span in spans)
        return "".join(out), spans

    def anonymize(self, text: str) -> str:
        """Anonymize a single string."""
        return self.scrub(text)[0]

    def sanitize(self, payload: Any, allow_numbers: bool = False) -> Any:
        """Recursively anonymize strings in dict/list payloads.

        Raw numbers are rejected by default because they can carry currency
        figures that no text rule can recognise; pass them as strings or set
        ``allow_numbers=True`` after review.
        """
        if isinstance(payload, str):
            return self.anonymize(payload)
        if isinstance(payload, Mapping):
            return {
                (self.anonymize(k) if isinstance(k, str) else k): self.sanitize(v, allow_numbers)
                for k, v in payload.items()
            }
        if isinstance(payload, (list, tuple, set)):
            return [self.sanitize(v, allow_numbers) for v in payload]
        if payload is None or isinstance(payload, bool):
            return payload
        if isinstance(payload, (int, float)) and allow_numbers:
            return payload
        raise TypeError(f"Refusing to egress unsupported/unsafe payload type: {type(payload).__name__}")

    # --------------------------------------------------------------- leak gate
    def assert_clean(self, scrubbed: str) -> None:
        """Raise ``LeakageError`` if sensitive material survives in ``scrubbed``."""
        stripped = _PLACEHOLDER_RE.sub(" ", scrubbed)
        problems = [label for label, pattern, _ in _RULES if pattern.search(stripped)]
        with self._lock:
            originals = list(self._vault.values())
        for original in originals:
            if len(original) >= 3 and re.search(
                r"(?<![\w\[])" + re.escape(original).replace(r"\ ", r"\s+") + r"(?![\w\]])",
                stripped,
                re.I,
            ):
                problems.append("VAULTED_ENTITY")
                break
        if problems:
            raise LeakageError(f"Egress blocked; residual sensitive categories: {sorted(set(problems))}")

    def egress_gate(self, payload: Any, allow_numbers: bool = False) -> Any:
        """Sanitize ``payload`` and verify it is clean. Only the return value may leave memory."""
        sanitized = self.sanitize(payload, allow_numbers)
        for text in self._iter_strings(sanitized):
            self.assert_clean(text)
        return sanitized

    @staticmethod
    def _iter_strings(obj: Any) -> Iterator[str]:
        if isinstance(obj, str):
            yield obj
        elif isinstance(obj, Mapping):
            for k, v in obj.items():
                yield from AirGappedProxy._iter_strings(k)
                yield from AirGappedProxy._iter_strings(v)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                yield from AirGappedProxy._iter_strings(v)

    # ----------------------------------------------------------------- utility
    def restore(self, text: str) -> str:
        """Re-identify placeholders locally (e.g. when rendering the final report)."""
        with self._lock:
            return _PLACEHOLDER_RE.sub(lambda m: self._vault.get(m.group(0), m.group(0)), text)

    def stats(self) -> Dict[str, int]:
        """Counts of replaced spans per label."""
        return dict(self._counts)

    def purge(self) -> None:
        """Wipe the vault, deny-list and counters from memory."""
        with self._lock:
            self._vault.clear()
            self._known.clear()
            self._alias.clear()
            self._counts.clear()
            self._deny_re = None

    def __repr__(self) -> str:  # never expose key or vault contents
        return f"AirGappedProxy(entities={len(self._vault)}, key=<redacted>)"