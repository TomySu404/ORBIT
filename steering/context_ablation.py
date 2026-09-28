"""
Extraction-context ablation (Q2): does the ROC gain come from the reasoning content,
or merely from reading after a long self-generated context?

Row 2 -> row 3 of the matched-control table changes the reasoning content of the
extraction context, but it also lengthens that context by hundreds of tokens. These
transforms interpolate between the two, each preserving the rollout's FINAL ANSWER
string (so the anchor token is unchanged) and, where applicable, its token count.

Modes
    full        identity -- the rollout as ROC produces it            (row 3)
    answer_only strip the reasoning, keep the answer                 (row 2)
    filler      reasoning replaced by task-neutral text, length matched   (C1)
    mismatched  reasoning replaced by another question's reasoning, length matched (C2)
    shuffled    own reasoning steps kept but permuted                     (C3)

Positives and negatives are transformed identically, so the pair count, the labels,
and the answer text are unchanged; only the extraction context varies.
"""
import re
from typing import List, Optional, Tuple

MODES = ("full", "answer_only", "filler", "mismatched", "shuffled")

# Markers that introduce the final answer, in priority order. The LAST match wins,
# so a rollout that reasons about "the answer is ..." mid-derivation still splits at
# its final statement.
_TAIL_PATTERNS = [
    r"####\s*.*$",
    r"[Tt]he answer is\b.*$",
    r"[Ff]inal [Aa]nswer\s*[:.]?.*$",
    r"[Aa]nswer\s*[:]\s*.*$",
    r"\\boxed\{.*$",
]

_FILLER_SENTENCES = [
    "We now proceed to the next part of the response.",
    "This section continues the response in the same format.",
    "The following text is included to occupy the same space.",
    "No additional information is conveyed by this passage.",
]


def split_reasoning_tail(text: str) -> Tuple[str, str]:
    """Split a rollout into (reasoning, tail).

    `tail` is the span that states the final answer and is preserved verbatim by every
    transform, so the anchor token (last token of the sequence) never changes.
    Returns ("", text) when no reasoning can be identified -- a short rollout, which is
    the near-empty-trajectory case the paper discusses for classification tasks.
    """
    best = None
    for pat in _TAIL_PATTERNS:
        for m in re.finditer(pat, text, flags=re.MULTILINE):
            if best is None or m.start() > best:
                best = m.start()
    if best is not None and best > 0:
        return text[:best], text[best:]

    # No explicit marker: fall back to the last non-empty line.
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) >= 2:
        tail = lines[-1]
        idx = text.rfind(tail)
        return text[:idx], text[idx:]
    return "", text


def _ntok(tokenizer, s: str) -> int:
    if not s:
        return 0
    return len(tokenizer(s, add_special_tokens=False)["input_ids"])


def _fit_to_tokens(tokenizer, s: str, target: int) -> str:
    """Truncate or repeat `s` so it occupies about `target` tokens."""
    if target <= 0:
        return ""
    ids = tokenizer(s, add_special_tokens=False)["input_ids"]
    if not ids:
        return ""
    while len(ids) < target:
        ids = ids + ids
    return tokenizer.decode(ids[:target], skip_special_tokens=True)


def _split_steps(reasoning: str) -> List[str]:
    """Split reasoning into steps at line breaks, falling back to sentences."""
    parts = [p for p in reasoning.split("\n") if p.strip()]
    if len(parts) >= 2:
        return parts
    parts = re.split(r"(?<=[.!?])\s+", reasoning.strip())
    return [p for p in parts if p.strip()]


def transform(
    mode: str,
    text: str,
    tokenizer,
    rng,
    donor: Optional[str] = None,
) -> str:
    """Apply one extraction-context transform to a single rollout.

    Args:
        mode: one of MODES.
        text: the rollout to transform.
        tokenizer: used only for token-count matching.
        rng: random.Random, for `shuffled` and for donor truncation offsets.
        donor: another question's rollout; required for `mismatched`.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if mode == "full":
        return text

    reasoning, tail = split_reasoning_tail(text)
    if not reasoning.strip():
        # Nothing to ablate -- every mode degenerates to the original text.
        return text

    if mode == "answer_only":
        return tail.lstrip()

    target = _ntok(tokenizer, reasoning)

    if mode == "filler":
        filler_src = " ".join(_FILLER_SENTENCES) + " "
        new_reasoning = _fit_to_tokens(tokenizer, filler_src, target)

    elif mode == "mismatched":
        if not donor:
            return text
        donor_reasoning, _ = split_reasoning_tail(donor)
        if not donor_reasoning.strip():
            donor_reasoning = donor
        new_reasoning = _fit_to_tokens(tokenizer, donor_reasoning, target)

    elif mode == "shuffled":
        steps = _split_steps(reasoning)
        if len(steps) < 2:
            return text
        order = list(range(len(steps)))
        # Guarantee the permutation actually moves something.
        for _ in range(8):
            rng.shuffle(order)
            if order != sorted(order):
                break
        new_reasoning = "\n".join(steps[i] for i in order)

    else:  # pragma: no cover
        return text

    sep = "\n" if not new_reasoning.endswith("\n") else ""
    return f"{new_reasoning}{sep}{tail.lstrip()}"


def describe(mode: str) -> str:
    return {
        "full": "full own trajectory (ROC, row 3)",
        "answer_only": "answer only, matched TF (row 2)",
        "filler": "C1 length-matched neutral filler",
        "mismatched": "C2 another question's reasoning, length matched",
        "shuffled": "C3 own reasoning steps permuted",
    }[mode]
