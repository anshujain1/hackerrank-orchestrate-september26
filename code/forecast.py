from __future__ import annotations
from calendar import monthrange
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

from models import (
    Direction,
    EventStatus,
    FinancialFact,
    ForecastPoint,
    ForecastResult,
    RecurringSeries,
)


# ---------------------------------------------------------------------------
# Internal flow representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CashFlow:
    """
    A single cash movement used by the forecast engine.

    fact_id is the source event/series fact when available.
    Synthetic recurring flows use a deterministic synthetic identifier.
    """

    on_date: date
    amount: Decimal
    fact_id: Optional[str]
    reason: str
    category: Optional[str] = None
    direction: Optional[Direction] = None
    synthetic: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signed_amount(
    amount: Decimal,
    direction: Direction,
) -> Decimal:
    if direction == Direction.CREDIT:
        return amount

    if direction == Direction.DEBIT:
        return -amount

    raise ValueError(
        f"Cannot create cash flow from direction={direction!r}"
    )


def _series_amount_on(
    series: RecurringSeries,
    occurrence_date: date,
) -> Optional[Decimal]:
    """
    Resolve the recurring amount applicable on a particular date.

    Amount overrides are applied chronologically. The latest applicable
    override wins.
    """

    amount = series.typical_amount

    if amount is None:
        return None

    for override in sorted(
        series.amount_overrides,
        key=lambda item: item.effective_date,
    ):
        if occurrence_date >= override.effective_date:
            amount = override.amount

    if amount is None or amount < 0:
        return None

    return amount

def _next_month_same_day(d: date) -> date:
    if d.month == 12:
        year = d.year + 1
        month = 1
    else:
        year = d.year
        month = d.month + 1

    day = min(d.day, monthrange(year, month)[1])
    return date(year, month, day)

def _occurrence_dates(series, horizon_start, horizon_end):
    current = series.anchor_date

    while True:
        if series.interval_days == 30:
            current = _next_month_same_day(current)
        else:
            current += timedelta(days=series.interval_days)

        if current > horizon_end:
            break

        if current >= horizon_start:
            yield current


# ---------------------------------------------------------------------------
# Known concrete flows
# ---------------------------------------------------------------------------

def known_flows(
    facts: list[FinancialFact],
    horizon_start: date,
    horizon_end: date,
) -> list[CashFlow]:
    """
    Extract concrete future cash movements already represented by the data.

    Included:
        - scheduled credits/debits
        - pending debits

    Excluded:
        - settled historical events
        - pending credits
        - cancelled events
        - failed events
        - unrealized events
        - non-cash events
        - facts excluded by canonicalization
        - unresolved blank-amount facts

    Pending credits are deliberately excluded because they are not yet
    settled and therefore cannot safely increase available funds.
    """

    if horizon_start > horizon_end:
        return []

    flows: list[CashFlow] = []

    for fact in facts:
        if not fact.counts_in_forecast:
            continue

        if fact.home_currency_amount is None:
            continue

        if fact.status in {
            EventStatus.CANCELLED,
            EventStatus.FAILED,
            EventStatus.UNREALIZED,
        }:
            continue

        if fact.direction == Direction.NON_CASH:
            continue

        # Scheduled events are known future cash movements.
        #
        # Pending debits are treated as expected outflows because the
        # affordability engine must reserve for them.
        #
        # Pending credits are excluded because they are not settled.
        is_future_cash_event = (
            fact.status == EventStatus.SCHEDULED
            or (
                fact.status == EventStatus.PENDING
                and fact.direction == Direction.DEBIT
            )
        )

        if not is_future_cash_event:
            continue

        if not (
            horizon_start
            <= fact.date
            <= horizon_end
        ):
            continue

        flows.append(
            CashFlow(
                on_date=fact.date,
                amount=_signed_amount(
                    fact.home_currency_amount,
                    fact.direction,
                ),
                fact_id=fact.event_id,
                reason=f"known {fact.category}",
                category=fact.category,
                direction=fact.direction,
                synthetic=False,
            )
        )

    return flows


# ---------------------------------------------------------------------------
# Recurring series projection
# ---------------------------------------------------------------------------

def _known_occurrence_keys(
    facts: list[FinancialFact],
    horizon_start: date,
    horizon_end: date,
) -> set[tuple[date, str, Direction]]:
    """
    Build keys for concrete future events.

    A recurring series must not create a synthetic occurrence when the
    dataset already contains a concrete future event for the same
    date/category/direction.
    """

    keys: set[tuple[date, str, Direction]] = set()

    for fact in facts:
        if not fact.counts_in_forecast:
            continue

        if fact.home_currency_amount is None:
            continue

        if fact.direction == Direction.NON_CASH:
            continue

        if fact.status not in {
            EventStatus.SCHEDULED,
            EventStatus.PENDING,
        }:
            continue

        if not (
            horizon_start
            <= fact.date
            <= horizon_end
        ):
            continue

        keys.add(
            (
                fact.date,
                fact.category,
                fact.direction,
            )
        )

    return keys


def project_series(
    series_map: dict[tuple, RecurringSeries],
    horizon_start: date,
    horizon_end: date,
    facts: Optional[list[FinancialFact]] = None,
) -> list[CashFlow]:
    """
    Generate synthetic future recurring cash flows.

    A synthetic occurrence is skipped when a concrete future event already
    represents the same category/direction/date.

    This prevents recurrence reconstruction from double-counting explicit
    future transactions.
    """

    if horizon_start > horizon_end:
        return []

    facts = facts or []

    known_keys = _known_occurrence_keys(
        facts,
        horizon_start,
        horizon_end,
    )

    flows: list[CashFlow] = []

    for series_id, series in sorted(
        series_map.items(),
        key=lambda item: item[0],
    ):
        if series.interval_days <= 0:
            continue

        direction = series.direction

        if direction not in {
            Direction.CREDIT,
            Direction.DEBIT,
        }:
            continue

        sign = (
            Decimal("1")
            if direction == Direction.CREDIT
            else Decimal("-1")
        )

        for occurrence_date in _occurrence_dates(
            series,
            horizon_start,
            horizon_end,
        ):
            # A termination date means the recurring series stops after
            # that date. The occurrence on the termination date itself
            # remains valid.
            if (
                series.terminated_after is not None
                and occurrence_date > series.terminated_after
            ):
                break

            amount = _series_amount_on(
                series,
                occurrence_date,
            )

            if amount is None:
                continue

            if amount < 0:
                continue

            key = (
                occurrence_date,
                series.category,
                direction,
            )

            if key in known_keys:
                continue

            flows.append(
                CashFlow(
                    on_date=occurrence_date,
                    amount=sign * amount,
                    fact_id=(
                        f"series:{series_id}:"
                        f"{occurrence_date.isoformat()}"
                    ),
                    reason=f"projected {series.category}",
                    category=series.category,
                    direction=direction,
                    synthetic=True,
                )
            )

    return flows


# ---------------------------------------------------------------------------
# Combined flow construction
# ---------------------------------------------------------------------------

def build_flows(
    facts: list[FinancialFact],
    series_map: dict[tuple, RecurringSeries],
    horizon_start: date,
    horizon_end: date,
) -> list[CashFlow]:
    """
    Build the complete future cash-flow stream.

    Concrete known events take precedence over synthetic recurring
    projections.
    """

    known = known_flows(
        facts,
        horizon_start,
        horizon_end,
    )

    projected = project_series(
        series_map,
        horizon_start,
        horizon_end,
        facts=facts,
    )

    return sorted(
        [*known, *projected],
        key=lambda flow: (
            flow.on_date,
            flow.synthetic,
            flow.fact_id or "",
        ),
    )


# ---------------------------------------------------------------------------
# Forecast engine
# ---------------------------------------------------------------------------

def forecast(
    starting_balance: Decimal,
    flows: list[CashFlow],
    horizon_start: date,
    horizon_end: date,
    min_balance_to_keep: Optional[Decimal] = None,
) -> ForecastResult:
    """
    Simulate the cash balance across the forecast horizon.

    Same-day flows are NETTED before evaluating the day's ending balance.
    This avoids inventing an intraday ordering that the dataset does not
    provide.
    """

    if horizon_start > horizon_end:
        raise ValueError(
            "horizon_start cannot be after horizon_end"
        )

    balance = starting_balance
    lowest_balance = starting_balance
    lowest_date = horizon_start

    # Aggregate all flows occurring on the same date.
    daily_flows: dict[date, list[CashFlow]] = defaultdict(list)

    for flow in flows:
        if not (
            horizon_start
            <= flow.on_date
            <= horizon_end
        ):
            continue

        daily_flows[flow.on_date].append(flow)

    points: list[ForecastPoint] = []

    for flow_date in sorted(daily_flows):
        day_flows = daily_flows[flow_date]

        daily_delta = sum(
            (flow.amount for flow in day_flows),
            Decimal("0"),
        )

        balance += daily_delta

        # Keep one ForecastPoint per underlying flow so downstream
        # explanations can still identify what caused the movement.
        #
        # Every point for a date carries the same end-of-day balance
        # because intraday ordering is intentionally not assumed.
        for flow in day_flows:
            points.append(
                ForecastPoint(
                    on_date=flow_date,
                    delta=flow.amount,
                    balance_after=balance,
                    fact_id=flow.fact_id,
                    reason=flow.reason,
                )
            )

        if balance < lowest_balance:
            lowest_balance = balance
            lowest_date = flow_date

    breached = (
        min_balance_to_keep is not None
        and lowest_balance < min_balance_to_keep
    )

    return ForecastResult(
        points=tuple(points),
        min_balance=lowest_balance,
        min_balance_date=lowest_date,
        breached=breached,
    )


# ---------------------------------------------------------------------------
# Minimum-balance helper
# ---------------------------------------------------------------------------

def min_balance_over(
    starting_balance: Decimal,
    flows: list[CashFlow],
    start: date,
    end: date,
) -> Decimal:
    """
    Return the lowest end-of-day balance over [start, end].

    Flows outside the requested interval are ignored.

    Same-day flows are netted before evaluating that day's balance.
    """

    if start > end:
        raise ValueError(
            "start cannot be after end"
        )

    daily_delta: dict[date, Decimal] = defaultdict(
        lambda: Decimal("0")
    )

    for flow in flows:
        if start <= flow.on_date <= end:
            daily_delta[flow.on_date] += flow.amount

    balance = starting_balance
    lowest = starting_balance

    for flow_date in sorted(daily_delta):
        balance += daily_delta[flow_date]
        lowest = min(
            lowest,
            balance,
        )

    return lowest