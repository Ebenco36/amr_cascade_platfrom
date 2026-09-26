"""String formats of the raw ARMD-family extracts, detected and re-rendered.

The three sites write the same information in different spellings: timestamps
("2009-11-27 02:52:00+00:00", "2017-08-14 22:40:00 UTC",
"2020-09-11T06:42:00.000Z", "2008-01-16 18:30:00"), missing values ("Null" or
an empty field), integers written as floats ("1949.0"), and float32 round-trip
artefacts ("98.4000015258789"). The replica profiler records which spelling each
column uses and the replica generator writes values back in that spelling, so
the synthetic files exercise the same parsing paths as the real extracts.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd

# key -> (strftime pattern, full-match regex)
TIME_FORMATS: dict[str, tuple[str, str]] = {
    "space_offset": ("%Y-%m-%d %H:%M:%S+00:00", r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+00:00"),
    "space_utc": ("%Y-%m-%d %H:%M:%S UTC", r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC"),
    "t_millis_z": ("%Y-%m-%dT%H:%M:%S.000Z", r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"),
    "space_naive": ("%Y-%m-%d %H:%M:%S", r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"),
}

MISSING_TOKENS = ("Null", "NULL", "")

_INT_RE = re.compile(r"-?\d+")
_FLOAT_RE = re.compile(r"-?\d+\.\d+")


def detect_time_format(values: Iterable[str]) -> str | None:
    """Return the TIME_FORMATS key matching most non-missing values, or None."""
    counts: Counter[str] = Counter()
    for value in values:
        if value in MISSING_TOKENS:
            continue
        for key, (_, pattern) in TIME_FORMATS.items():
            if re.fullmatch(pattern, value):
                counts[key] += 1
                break
    return counts.most_common(1)[0][0] if counts else None


def render_times(timestamps: pd.Series, format_key: str) -> pd.Series:
    """Write naive UTC timestamps in one of the raw extracts' spellings."""
    pattern = TIME_FORMATS[format_key][0]
    return pd.Series(pd.DatetimeIndex(timestamps).strftime(pattern), index=timestamps.index, dtype="object")


def numeric_spec(values: pd.Series) -> dict[str, object]:
    """Describe how a numeric column is spelled.

    Returns ``kind`` ("int", "float" or "text"), the typical number of decimals,
    the share of integer-valued floats written with a trailing ".0", and the
    share of values that only reproduce through float32 (long artefact tails).
    """
    present = values[~values.isin(MISSING_TOKENS)].astype(str)
    if present.empty:
        return {"kind": "text", "decimals": 0, "float32_share": 0.0}
    sample = present.sample(n=min(len(present), 20_000), random_state=0) if len(present) > 20_000 else present
    is_int = sample.str.fullmatch(_INT_RE.pattern)
    is_float = sample.str.fullmatch(_FLOAT_RE.pattern)
    numeric_share = float((is_int | is_float).mean())
    if numeric_share < 0.95:
        return {"kind": "text", "decimals": 0, "float32_share": 0.0}
    if float(is_int.mean()) >= 0.999:
        return {"kind": "int", "decimals": 0, "float32_share": 0.0}
    decimals = sample[is_float].str.split(".").str[1].str.len()
    long_tail = decimals >= 8
    typical = int(decimals[~long_tail].median()) if (~long_tail).any() else 2
    return {
        "kind": "float",
        "decimals": max(1, min(typical, 4)),
        "float32_share": round(float(long_tail.mean()), 4),
    }


def render_numbers(
    values: np.ndarray,
    spec: dict[str, object],
    rng: np.random.Generator,
    missing_token: str,
) -> np.ndarray:
    """Render floats (NaN = missing) in the spelling ``spec`` describes."""
    out = np.full(len(values), missing_token, dtype=object)
    present = ~np.isnan(values)
    if not present.any():
        return out
    kind = spec.get("kind", "float")
    if kind == "int":
        out[present] = [str(int(v)) for v in np.rint(values[present])]
        return out
    decimals = int(spec.get("decimals", 2))
    rounded = np.round(values[present], decimals)
    float32_share = float(spec.get("float32_share", 0.0))
    use_float32 = rng.random(len(rounded)) < float32_share
    rendered = [
        repr(float(np.float32(v))) if f32 else repr(float(v))
        for v, f32 in zip(rounded, use_float32, strict=True)
    ]
    out[present] = rendered
    return out


def dominant_missing_token(token_counts: dict[str, int]) -> str:
    """Pick the spelling a column uses for missing values ("Null" when unseen)."""
    observed = {token: count for token, count in token_counts.items() if count > 0}
    if not observed:
        return "Null"
    return max(observed.items(), key=lambda item: item[1])[0]


def id_shape(value: str) -> str:
    """Map an identifier to its character-class shape: letters -> L, digits -> D."""
    return "".join("L" if ch.isalpha() else "D" if ch.isdigit() else ch for ch in value)


def generate_identifiers(
    count: int,
    shapes: list[tuple[str, float]],
    rng: np.random.Generator,
    *,
    used: set[str] | None = None,
) -> list[str]:
    """Create ``count`` unique synthetic identifiers with the raw shapes.

    Letter positions always start with "Z" and digit-only identifiers always
    start with "9", so a synthetic identifier is recognisable as synthetic while
    keeping the raw length and character layout.
    """
    if count <= 0:
        return []
    used = set() if used is None else used
    labels = [shape for shape, _ in shapes]
    weights = np.array([weight for _, weight in shapes], dtype=float)
    weights = weights / weights.sum()
    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    result: list[str] = []
    while len(result) < count:
        shape = labels[int(rng.choice(len(labels), p=weights))]
        chars: list[str] = []
        first_letter = True
        first_digit = True
        for position, symbol in enumerate(shape):
            if symbol == "L":
                chars.append("Z" if first_letter else str(rng.choice(letters)))
                first_letter = False
            elif symbol == "D":
                if first_digit and position == 0:
                    chars.append("9")
                else:
                    chars.append(str(int(rng.integers(0, 10))))
                first_digit = False
            else:
                chars.append(symbol)
        candidate = "".join(chars)
        if candidate in used:
            continue
        used.add(candidate)
        result.append(candidate)
    return result


def quantile_grid(values: pd.Series, points: int = 101, tail: float = 0.005) -> list[float]:
    """Quantiles of a numeric sample on an even grid (for inverse-CDF sampling).

    The grid runs from the ``tail`` to the ``1 - tail`` quantile rather than
    from the minimum to the maximum, so no stored value is an individual
    record's extreme (top- and bottom-coding).
    """
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return []
    grid = np.linspace(tail, 1.0 - tail, points)
    return [round(float(v), 6) for v in np.quantile(numeric.to_numpy(dtype=float), grid)]


def sample_from_quantiles(quantiles: list[float], size: int, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sampling with linear interpolation between stored quantiles."""
    if size <= 0:
        return np.empty(0, dtype=float)
    if not quantiles:
        return np.full(size, np.nan)
    grid = np.linspace(0.0, 1.0, len(quantiles))
    return np.interp(rng.random(size), grid, np.asarray(quantiles, dtype=float))


def suppressed_counts(counts: Counter | dict, min_count: int) -> dict[str, int]:
    """Drop categories below the small-cell threshold (disclosure control)."""
    return {str(key): int(value) for key, value in counts.items() if int(value) >= min_count}


def weighted_choice(
    categories: list[str],
    weights: list[float] | np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``size`` categories with probability proportional to ``weights``."""
    if size <= 0:
        return np.empty(0, dtype=object)
    probabilities = np.asarray(weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    index = rng.choice(len(categories), size=size, p=probabilities)
    return np.asarray(categories, dtype=object)[index]


def histogram_draw(histogram: dict[str, int], size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw integers from a {value: count} histogram."""
    values = np.array([int(key) for key in histogram], dtype=int)
    weights = np.array(list(histogram.values()), dtype=float)
    if size <= 0 or values.size == 0:
        return np.zeros(max(size, 0), dtype=int)
    return rng.choice(values, size=size, p=weights / weights.sum())
