from __future__ import annotations

import hashlib
import re
import statistics
from datetime import date
from decimal import Decimal
from typing import Optional

from models import (
    AmountOverride,
    Direction,
    EventStatus,
    FinancialFact,
    RecurringSeries,
)
from canonicalize import KNOWN_RECURRING_CATEGORIES


RECURRING_MIN_OCCURRENCES = 3
RECURRING_MAX_RELATIVE_STD = 0.45

DEFAULT_CADENCE_DAYS = {
    "salary": 30,
    "rent": 30,
    "utilities": 30,
    "education": 30,
    "debt_repayment": 30,
    "music_subscription": 30,
    "delivery_membership": 30,
    "housing": 30,
    "insurance": 30,
    "family_support": 30,
    "work_expense": 30,
    "gym": 30,
    "streaming": 30,
    "cloud_storage": 30,
    "groceries": 7,
    "transport": 7,
    "dining": 14,
    "healthcare": 30,
    "entertainment": 14,
}


def _normalize_description(description: str) -> str:
    """
    Normalize descriptions conservatively so small formatting differences
    do not split an otherwise identical recurring series.
    """
    text = (description or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _series_key(
    fact: FinancialFact,
) -> tuple[str, Direction]:
    """
    Candidate recurring-series identity.

    Recurring behavior is primarily defined by financial category and
    direction. Descriptions may vary across merchants or transaction
    descriptions within the same recurring category.
    """
    return (
        fact.category,
        fact.direction,
    )


def _positive_amounts(
    facts: list[FinancialFact],
) -> list[Decimal]:
    return [
        fact.home_currency_amount
        for fact in facts
        if (
            fact.home_currency_amount is not None
            and fact.home_currency_amount > 0
        )
    ]


def _regular_cadence(
    dates: list[date],
) -> tuple[Optional[int], bool]:
    """
    Return (cadence, is_regular).

    `is_regular` tells us whether the observed dates have a stable enough
    interval to trust the observed cadence.
    """
    unique_dates = sorted(set(dates))

    if len(unique_dates) < 2:
        return None, False

    diffs = [
        (b - a).days
        for a, b in zip(unique_dates, unique_dates[1:])
        if (b - a).days > 0
    ]

    if not diffs:
        return None, False

    median_diff = statistics.median(diffs)

    if median_diff <= 0:
        return None, False

    # Relative dispersion is measured around the median rather than mean.
    deviations = [
        abs(diff - median_diff)
        for diff in diffs
    ]

    median_deviation = statistics.median(deviations)

    relative_deviation = (
        Decimal(str(median_deviation))
        / Decimal(str(median_diff))
    )

    is_regular = relative_deviation <= Decimal("0.45")

    return round(median_diff), is_regular


def _is_recurring(
    category: str,
    dates: list[date],
) -> bool:
    """
    Determine whether observations plausibly form a recurring series.

    Known recurring categories may form a series from a single observation,
    because the dataset's category semantics already identify them as
    recurring.

    Unknown categories require at least three observations and reasonably
    stable spacing.
    """
    unique_dates = sorted(set(dates))

    if category in KNOWN_RECURRING_CATEGORIES:
        return len(unique_dates) >= 1

    if len(unique_dates) < RECURRING_MIN_OCCURRENCES:
        return False

    cadence, regular = _regular_cadence(unique_dates)

    return cadence is not None and regular


def _cadence(
    category: str,
    dates: list[date],
) -> Optional[int]:
    """
    Select a conservative recurrence cadence.

    Known recurring categories use their category-level default unless
    there is enough repeated evidence to confidently establish a regular
    observed cadence.
    """
    observed_cadence, regular = _regular_cadence(dates)

    default_cadence = DEFAULT_CADENCE_DAYS.get(category)

    # Known categories should not invent long cadences from sparse data.
    if default_cadence is not None:
        if len(set(dates)) >= 4 and observed_cadence is not None and regular:
            return observed_cadence
        return default_cadence

    return observed_cadence

def _typical_amount(
    facts: list[FinancialFact],
) -> Optional[Decimal]:
    """
    Use the median historical amount as the base recurring amount.

    This is more robust than using the latest occurrence, which could be
    an unusual spike.
    """
    amounts = _positive_amounts(facts)

    if not amounts:
        return None

    return Decimal(
        str(statistics.median(amounts))
    )


def _latest_valid_fact(
    facts: list[FinancialFact],
) -> Optional[FinancialFact]:
    valid = [
        fact
        for fact in facts
        if fact.status not in {
            EventStatus.CANCELLED,
            EventStatus.FAILED,
            EventStatus.UNREALIZED,
        }
    ]

    if not valid:
        return None

    return max(
        valid,
        key=lambda fact: (
            fact.date,
            fact.settlement_date or fact.date,
            fact.event_id,
        ),
    )


def _latest_settled_fact(
    facts: list[FinancialFact],
) -> Optional[FinancialFact]:
    settled = [
        fact
        for fact in facts
        if (
            fact.status == EventStatus.SETTLED
            and fact.home_currency_amount is not None
        )
    ]

    if not settled:
        return None

    return max(
        settled,
        key=lambda fact: (
            fact.date,
            fact.settlement_date or fact.date,
            fact.event_id,
        ),
    )


def _dedupe_member_facts(
    facts: list[FinancialFact],
) -> list[FinancialFact]:
    """
    Defensive event-id deduplication.

    Explicit evidence-based duplicate handling belongs in canonicalize.py;
    this is only a safety net against accidental repeated input.
    """
    seen: set[str] = set()
    result: list[FinancialFact] = []

    for fact in sorted(
        facts,
        key=lambda item: (
            item.date,
            item.event_id,
        ),
    ):
        if fact.event_id in seen:
            continue

        seen.add(fact.event_id)
        result.append(fact)

    return result


def _applicable_overrides(
    member_ids: set[str],
    series_overrides: list[tuple[str, AmountOverride]],
) -> tuple[AmountOverride, ...]:
    overrides = [
        override
        for event_id, override in series_overrides
        if event_id in member_ids
    ]

    return tuple(
        sorted(
            overrides,
            key=lambda override: (
                override.effective_date,
                override.amount,
                override.currency,
            ),
        )
    )


def _termination_date(
    member_ids: set[str],
    series_terminations: list[tuple[str, date]],
) -> Optional[date]:
    dates = [
        termination_date
        for event_id, termination_date in series_terminations
        if event_id in member_ids
    ]

    return min(dates) if dates else None


def _stable_series_suffix(
    category: str,
    direction: Direction,
) -> str:
    raw = (
        f"{category}|"
        f"{direction.value}"
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()[:12]


def build_series(
    user_id: str,
    facts: list[FinancialFact],
    series_overrides: list[tuple[str, AmountOverride]],
    series_terminations: list[tuple[str, date]],
    home_currency: str,
) -> dict[tuple, RecurringSeries]:
    """
    Reconstruct recurring financial series from canonical facts.

    canonicalize.py:
        determines what each financial fact means.

    series.py:
        determines which facts form recurring patterns.

    forecast.py:
        determines future cash-flow consequences.

    No affordability decision is made here.
    """

    groups: dict[
        tuple[str, Direction],
        list[FinancialFact],
        ] = {}
    # ------------------------------------------------------------------
    # 1. Build candidate groups
    # ------------------------------------------------------------------

    for fact in facts:
        if fact.status in {
            EventStatus.CANCELLED,
            EventStatus.FAILED,
            EventStatus.UNREALIZED,
        }:
            continue

        if fact.home_currency_amount is None:
            # Never infer a recurring cash amount from unresolved data.
            continue

        if fact.direction not in {
            Direction.CREDIT,
            Direction.DEBIT,
        }:
            continue

        groups.setdefault(
            _series_key(fact),
            [],
        ).append(fact)

    series_map: dict[
        tuple,
        RecurringSeries,
    ] = {}

    # ------------------------------------------------------------------
    # 2. Reconstruct each candidate series
    # ------------------------------------------------------------------

    for key, raw_group in groups.items():
        group = _dedupe_member_facts(raw_group)

        if not group:
            continue

        category, direction = key

        cadence_dates = [
            fact.date
            for fact in group
            if fact.status in {
                EventStatus.SETTLED,
                EventStatus.SCHEDULED,
            }
        ]

        if not _is_recurring(
            category,
            cadence_dates,
        ):
            continue

        cadence = _cadence(
            category,
            cadence_dates,
        )

        if cadence is None or cadence <= 0:
            continue

        # ----------------------------------------------------------------
        # 3. Representative recurring amount
        # ----------------------------------------------------------------

        typical_amount = _typical_amount(group)

        if typical_amount is None:
            continue

        # ----------------------------------------------------------------
        # 4. Forecast anchor
        #
        # Latest valid concrete occurrence is the starting point for
        # future synthetic occurrences.
        # ----------------------------------------------------------------

        latest_fact = _latest_valid_fact(group)

        if latest_fact is None:
            continue

        anchor_date = latest_fact.date

        # ----------------------------------------------------------------
        # 5. Provenance
        # ----------------------------------------------------------------

        member_ids = tuple(
            fact.event_id
            for fact in group
        )

        member_id_set = set(member_ids)

        overrides = _applicable_overrides(
            member_id_set,
            series_overrides,
        )

        terminated_after = _termination_date(
            member_id_set,
            series_terminations,
        )

        # ----------------------------------------------------------------
        # 6. Flexibility
        #
        # Prefer the latest settled fact because it reflects an actual
        # realized financial state.
        # ----------------------------------------------------------------

        base_fact = (
            _latest_settled_fact(group)
            or latest_fact
        )

        # ----------------------------------------------------------------
        # 7. Stable deterministic series ID
        # ----------------------------------------------------------------

        suffix = _stable_series_suffix(
                category,
                direction
                        )

        series_id = (
            f"series_{user_id}_"
            f"{category}_"
            f"{direction.value}_"
            f"{suffix}"
        )

        series_map[key] = RecurringSeries(
            series_id=series_id,
            user_id=user_id,
            category=category,
            direction=direction,
            typical_amount=typical_amount,
            currency=home_currency,
            interval_days=cadence,
            anchor_date=anchor_date,
            flexibility=base_fact.flexibility,
            minimum_allowed_amount=base_fact.minimum_allowed_amount,
            member_fact_ids=member_ids,
            amount_overrides=overrides,
            terminated_after=terminated_after,
        )

    return series_map