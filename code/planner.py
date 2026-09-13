from __future__ import annotations
from calendar import monthrange
from datetime import date, timedelta

from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from itertools import combinations
from typing import Optional

from models import (
    AffordabilityStatus,
    Decision,
    PaymentMethod,
    PaymentOption,
    PlanCandidate,
    PlanPayment,
    Request,
    SpendingChange,
    SpendingChangeAction,
    UserProfile,
)
from forecast import CashFlow, build_flows


TWO_PLACES = Decimal("0.01")


def q(value: Decimal) -> Decimal:
    return value.quantize(TWO_PLACES)


# ---------------------------------------------------------------------------
# CashFlow helpers
# ---------------------------------------------------------------------------

def _flow_parts(
    flow: CashFlow,
) -> tuple[date, Decimal, str, bool, str]:
    """
    Return the planner-relevant fields from a forecast CashFlow.

    Planner deliberately consumes CashFlow objects rather than relying on
    tuple positions. Forecast.py owns the CashFlow representation.
    """
    return (
        flow.on_date,
        flow.amount,
        flow.category,
        flow.synthetic,
        flow.fact_id,
    )


def _payment_flow(
    payment_date: date,
    amount: Decimal,
    identifier: str,
) -> CashFlow:
    """
    Construct a synthetic cash flow representing a payment.
    """
    
    return CashFlow(
        on_date=payment_date,
        amount=-amount,
        category="request_payment",
        synthetic=True,
        fact_id=identifier,
        reason="request payment",
    )


# ---------------------------------------------------------------------------
# Spending-change candidate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpendingChangeCandidate:
    """
    A recurring expense that the user has explicitly allowed us to change.

    event_id:
        Canonical/latest real event ID exposed in the final decision.

    series_id:
        Recurring-series identity used internally to modify forecast flows.
    """

    series_id: str
    event_id: str
    occurrence_dates: tuple[date, ...]
    stoppable: bool
    reducible: bool
    minimum_allowed_amount: Optional[Decimal]
    typical_amount: Decimal

    @property
    def benefit_if_stopped(self) -> Decimal:
        return self.typical_amount * len(self.occurrence_dates)

    @property
    def benefit_if_reduced(self) -> Decimal:
        if self.minimum_allowed_amount is None:
            return Decimal("0")

        saving = max(
            Decimal("0"),
            self.typical_amount - self.minimum_allowed_amount,
        )

        return saving * len(self.occurrence_dates)


def _occurrence_dates(
    series,
    horizon_start: date,
    horizon_end: date,
) -> tuple[date, ...]:
    """
    Generate recurring occurrences inside the planning horizon.

    Keep monthly recurrence aligned with forecast.py:
    interval_days == 30 means calendar-month recurrence, not
    repeated 30-day jumps.
    """

    dates: list[date] = []

    if series.interval_days <= 0:
        return ()

    current = series.anchor_date

    while True:
        if series.interval_days in (30, 31):
            year = current.year + (current.month // 12)
            month = current.month % 12 + 1

            day = min(
                series.anchor_date.day,
                monthrange(year, month)[1],
            )

            current = date(year, month, day)
        else:
            current += timedelta(days=series.interval_days)

        if current > horizon_end:
            break

        if current >= horizon_start:
            if (
                series.terminated_after is None
                or current <= series.terminated_after
            ):
                dates.append(current)

    return tuple(dates)

  


def build_spending_candidates(
    series_map: dict,
    profile: UserProfile,
    horizon_start: date,
    horizon_end: date,
) -> list[SpendingChangeCandidate]:
    """
    Build only spending changes explicitly permitted by the profile.

    Protected categories always win over reduce/stop preferences.
    """

    stop_categories = {
    category.lower()
    for category in profile.expense_categories_user_is_willing_to_stop
}

    reduce_categories = {
    category.lower()
    for category in profile.expense_categories_user_is_willing_to_reduce
}

    protected_categories = {
        category.lower()
        for category in profile.expense_categories_to_protect
    }

    candidates: list[SpendingChangeCandidate] = []

    for series in series_map.values():
        if series.direction.value != "debit":
            continue

        category = series.category.lower()

        if category in protected_categories:
            continue

        stoppable = category in stop_categories

        reducible = (
            category in reduce_categories
            and series.minimum_allowed_amount is not None
        )

        if not stoppable and not reducible:
            continue

        dates = _occurrence_dates(
            series,
            horizon_start,
            horizon_end,
        )

        if not dates:
            continue

        # Final SpendingChange must point to a real canonical event,
        # never to a synthetic forecast occurrence.
        event_id = (
            series.member_fact_ids[-1]
            if series.member_fact_ids
            else series.series_id
        )

        candidates.append(
            SpendingChangeCandidate(
                series_id=series.series_id,
                event_id=event_id,
                occurrence_dates=dates,
                stoppable=stoppable,
                reducible=reducible,
                minimum_allowed_amount=series.minimum_allowed_amount,
                typical_amount=series.typical_amount,
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.series_id,
            candidate.event_id,
        ),
    )


# ---------------------------------------------------------------------------
# Checkpoint / balance calculations
# ---------------------------------------------------------------------------

def build_checkpoints(
    starting_balance: Decimal,
    flows: list[CashFlow],
    horizon_end: date,
) -> list[tuple[date, Decimal]]:
    """
    Convert CashFlow objects into cumulative end-of-day balances.

    Multiple flows on the same day are netted together.
    """

    by_date: dict[date, Decimal] = {}

    for flow in flows:
        flow_date, amount, _, _, _ = _flow_parts(flow)

        if flow_date > horizon_end:
            continue

        by_date[flow_date] = (
            by_date.get(flow_date, Decimal("0"))
            + amount
        )

    running = starting_balance
    checkpoints: list[tuple[date, Decimal]] = []

    for flow_date in sorted(by_date):
        running += by_date[flow_date]
        checkpoints.append(
            (flow_date, running)
        )

    return checkpoints


def balance_before(
    target_date: date,
    checkpoints: list[tuple[date, Decimal]],
    starting_balance: Decimal,
) -> Decimal:
    """
    Balance immediately before target_date's flows.
    """

    balance = starting_balance

    for checkpoint_date, checkpoint_balance in checkpoints:
        if checkpoint_date < target_date:
            balance = checkpoint_balance
        else:
            break

    return balance


def suffix_min_from(
    target_date: date,
    checkpoints: list[tuple[date, Decimal]],
    starting_balance: Decimal,
    horizon_end: date,
) -> Decimal:
    """
    Lowest end-of-day balance from target_date through horizon_end.
    """

    balance = balance_before(
        target_date,
        checkpoints,
        starting_balance,
    )

    lowest = balance

    for checkpoint_date, checkpoint_balance in checkpoints:
        if checkpoint_date < target_date:
            continue

        if checkpoint_date > horizon_end:
            break

        balance = checkpoint_balance
        lowest = min(lowest, balance)

    return lowest


def natural_min(
    starting_balance: Decimal,
    checkpoints: list[tuple[date, Decimal]],
    horizon_start: date,
    horizon_end: date,
) -> Decimal:
    return suffix_min_from(
        horizon_start,
        checkpoints,
        starting_balance,
        horizon_end,
    )


def compute_amount_safe_to_pay(
    starting_balance: Decimal,
    min_balance_to_keep: Decimal,
    checkpoints: list[tuple[date, Decimal]],
    horizon_start: date,
    horizon_end: date,
    requested_amount: Decimal,
) -> Decimal:
    """
    Baseline-only amount safe to pay.

    Spending changes MUST NOT inflate this value.
    """

    lowest_natural_balance = natural_min(
        starting_balance,
        checkpoints,
        horizon_start,
        horizon_end,
    )

    headroom = (
        lowest_natural_balance
        - min_balance_to_keep
    )

    return max(
        Decimal("0"),
        min(
            requested_amount,
            q(headroom),
        ),
    )


def compute_earliest_full_payment_date(
    starting_balance: Decimal,
    min_balance_to_keep: Decimal,
    checkpoints: list[tuple[date, Decimal]],
    horizon_start: date,
    horizon_end: date,
    requested_amount: Decimal,
) -> Optional[date]:
    """
    Baseline-only earliest date on which the full requested amount can
    safely be paid while preserving the minimum balance.
    """

    current = horizon_start

    while current <= horizon_end:
        lowest_after_payment = suffix_min_from(
            current,
            checkpoints,
            starting_balance,
            horizon_end,
        )

        if (
            lowest_after_payment - requested_amount
            >= min_balance_to_keep
        ):
            return current

        current += timedelta(days=1)

    return None


def full_payment_safe(
    starting_balance: Decimal,
    min_balance_to_keep: Decimal,
    checkpoints: list[tuple[date, Decimal]],
    horizon_start: date,
    horizon_end: date,
    requested_amount: Decimal,
) -> bool:
    lowest = natural_min(
        starting_balance,
        checkpoints,
        horizon_start,
        horizon_end,
    )

    return (
        lowest - requested_amount
        >= min_balance_to_keep
    )


# ---------------------------------------------------------------------------
# Applying spending changes
# ---------------------------------------------------------------------------

def _flow_matches_candidate(
    flow: CashFlow,
    candidate: SpendingChangeCandidate,
) -> bool:
    """
    Match a forecast flow to a spending-change candidate.

    Candidates may correspond either to:
      - concrete future events, identified by event_id, or
      - synthetic recurring occurrences, identified by series_id.
    """

    flow_date, _, _, synthetic, flow_id = _flow_parts(flow)

    if flow_date not in candidate.occurrence_dates:
        return False

    # Concrete future event: match its actual event ID.
    if not synthetic:
        return flow_id == candidate.event_id

    # Synthetic recurring occurrence: match its series.
    if flow_id == candidate.series_id:
        return True

    if isinstance(flow_id, str):
        return flow_id.startswith(
            f"series:{candidate.series_id}:"
        )

    return False


def apply_spending_changes(
    flows: list[CashFlow],
    changes: tuple[
        tuple[SpendingChangeCandidate, SpendingChangeAction],
        ...,
    ],
) -> list[CashFlow]:
    """
    Apply a complete intervention scenario.

    The baseline flow list is never mutated.
    """

    result = list(flows)

    for candidate, action in changes:

        if action == SpendingChangeAction.STOP:
            result = [
                flow
                for flow in result
                if not _flow_matches_candidate(
                    flow,
                    candidate,
                )
            ]

        elif action == SpendingChangeAction.REDUCE_TO:
            minimum = candidate.minimum_allowed_amount

            if minimum is None:
                continue

            reduced: list[CashFlow] = []

            for flow in result:
                if _flow_matches_candidate(
                    flow,
                    candidate,
                ):
                    reduced.append(
                        replace(
                            flow,
                            amount=-minimum,
                        )
                    )
                else:
                    reduced.append(flow)

            result = reduced

    return result


def make_spending_change(
    candidate: SpendingChangeCandidate,
    action: SpendingChangeAction,
) -> SpendingChange:

    if action == SpendingChangeAction.REDUCE_TO:
        return SpendingChange(
            action=action,
            event_id=candidate.event_id,
            new_amount=candidate.minimum_allowed_amount,
        )

    return SpendingChange(
        action=action,
        event_id=candidate.event_id,
    )


# ---------------------------------------------------------------------------
# Installment plans
# ---------------------------------------------------------------------------

def option_payment_dates(
    option: PaymentOption,
) -> list[date]:

    dates: list[date] = []

    for index in range(option.number_of_payments):
        dates.append(
            option.first_payment_date
            + timedelta(
                days=(
                    option.payment_frequency_days or 0
                ) * index
            )
        )

    return dates


def option_is_safe(
    option: PaymentOption,
    starting_balance: Decimal,
    min_balance_to_keep: Decimal,
    flows: list[CashFlow],
    horizon_start: date,
    horizon_end: date,
) -> tuple[bool, date, list[date]]:
    """
    Test the supplied installment option exactly as provided.

    We never alter:

      - payment amount
      - number of payments
      - payment frequency
      - financing fee
      - total payable
    """

    payment_dates = option_payment_dates(option)

    if not payment_dates:
        return (
            False,
            option.first_payment_date,
            [],
        )

    payment_flows = list(flows)

    for index, payment_date in enumerate(payment_dates):
        payment_flows.append(
            _payment_flow(
                payment_date,
                option.payment_amount,
                (
                    f"payment_option:"
                    f"{option.payment_option_id}:"
                    f"{index}"
                ),
            )
        )

    checkpoints = build_checkpoints(
        starting_balance,
        payment_flows,
        horizon_end,
    )

    lowest = natural_min(
        starting_balance,
        checkpoints,
        horizon_start,
        horizon_end,
    )

    return (
        lowest >= min_balance_to_keep,
        payment_dates[-1],
        payment_dates,
    )


# ---------------------------------------------------------------------------
# Plan ranking
# ---------------------------------------------------------------------------

def rank_key(candidate: PlanCandidate):
    """
    Challenge ranking:

    1. Complete by desired deadline
    2. No spending changes
    3. Lower total paid
    4. Earlier first payment
    5. Fewer payments
    6. Lowest payment_option_id
    """

    return (
        0 if candidate.completes_by_deadline else 1,
        1 if candidate.spending_changes else 0,
        candidate.total_paid,
        candidate.first_payment_date,
        candidate.num_payments,
        candidate.payment_option_id or "",
    )


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _eligible_plans(
    req: Request,
    profile: UserProfile,
    flows: list[CashFlow],
    options: list[PaymentOption],
    spending_changes: tuple[SpendingChange, ...],
    horizon_start: date,
    horizon_end: date,
) -> list[PlanCandidate]:
    """
    Generate full-payment and exact supplied-installment candidates
    for one financial world.
    """

    starting_balance = profile.current_available_balance
    minimum = profile.minimum_balance_to_keep
    requested_amount = req.requested_amount

    methods = {
        method.lower()
        for method in profile.payment_methods_user_will_consider
    }

    checkpoints = build_checkpoints(
        starting_balance,
        flows,
        horizon_end,
    )

    candidates: list[PlanCandidate] = []

    # ---------------------------------------------------------------
    # Full payment
    # ---------------------------------------------------------------

    if "full_payment" in methods:
        if full_payment_safe(
            starting_balance,
            minimum,
            checkpoints,
            horizon_start,
            horizon_end,
            requested_amount,
        ):
            candidates.append(
                PlanCandidate(
                    payment_method=PaymentMethod.FULL_PAYMENT,
                    payments=[
                        PlanPayment(
                            req.request_date,
                            requested_amount,
                        )
                    ],
                    spending_changes=list(
                        spending_changes
                    ),
                    completes_by_deadline=(
                        req.request_date
                        <= req.desired_completion_date
                    ),
                    total_paid=requested_amount,
                    first_payment_date=req.request_date,
                    num_payments=1,
                )
            )

    # ---------------------------------------------------------------
    # Exact installment options
    # ---------------------------------------------------------------

    if "installments" in methods:
        for option in sorted(
            options,
            key=lambda item: item.payment_option_id,
        ):
            if option.payment_method.lower() != "installments":
                continue

            safe, last_date, dates = option_is_safe(
                option,
                starting_balance,
                minimum,
                flows,
                horizon_start,
                horizon_end,
            )

            if not safe:
                continue

            candidates.append(
                PlanCandidate(
                    payment_method=PaymentMethod.INSTALLMENTS,
                    payments=[
                        PlanPayment(
                            payment_date,
                            option.payment_amount,
                        )
                        for payment_date in dates
                    ],
                    spending_changes=list(
                        spending_changes
                    ),
                    completes_by_deadline=(
                        last_date
                        <= req.desired_completion_date
                    ),
                    total_paid=option.total_payable_amount,
                    first_payment_date=dates[0],
                    num_payments=option.number_of_payments,
                    payment_option_id=option.payment_option_id,
                )
            )

    return candidates


# ---------------------------------------------------------------------------
# Partial payment
# ---------------------------------------------------------------------------

def build_partial_candidate(
    req: Request,
    profile: UserProfile,
    amount_safe_to_pay: Decimal,
    earliest_full_payment_date: Optional[date],
) -> Optional[PlanCandidate]:
    """
    Partial payment is exactly two payments:

      1. amount_safe_to_pay on request_date
      2. remainder on earliest baseline-safe full-payment date

    No spending changes are attached.
    """

    if not req.allows_partial_payment:
        return None

    methods = {
        method.lower()
        for method in profile.payment_methods_user_will_consider
    }

    if "partial_payment" not in methods:
        return None

    if not (
        Decimal("0")
        < amount_safe_to_pay
        < req.requested_amount
    ):
        return None

    if earliest_full_payment_date is None:
        return None

    if (
        earliest_full_payment_date
        > req.desired_completion_date
    ):
        return None

    remainder = q(
        req.requested_amount
        - amount_safe_to_pay
    )

    return PlanCandidate(
        payment_method=PaymentMethod.PARTIAL_PAYMENT,
        payments=[
            PlanPayment(
                req.request_date,
                amount_safe_to_pay,
            ),
            PlanPayment(
                earliest_full_payment_date,
                remainder,
            ),
        ],
        spending_changes=[],
        completes_by_deadline=True,
        total_paid=req.requested_amount,
        first_payment_date=req.request_date,
        num_payments=2,
    )


# ---------------------------------------------------------------------------
# Intervention scenarios
# ---------------------------------------------------------------------------

def _change_actions(
    candidate: SpendingChangeCandidate,
) -> tuple[SpendingChangeAction, ...]:

    actions: list[SpendingChangeAction] = []

    if candidate.stoppable:
        actions.append(
            SpendingChangeAction.STOP
        )

    if candidate.reducible:
        actions.append(
            SpendingChangeAction.REDUCE_TO
        )

    return tuple(actions)


def _build_intervention_scenarios(
    candidates: list[SpendingChangeCandidate],
    max_changes: int = 4,
):
    """
    Generate deterministic intervention combinations.

    This intentionally supports combinations such as:

        STOP event_A
        +
        REDUCE_TO event_B

    rather than using a greedy one-change strategy.
    """

    scenarios = [tuple()]

    candidates = sorted(
        candidates,
        key=lambda candidate: candidate.series_id,
    )

    max_size = min(
        max_changes,
        len(candidates),
    )

    for size in range(1, max_size + 1):

        for selected in combinations(
            candidates,
            size,
        ):
            action_lists = [
                _change_actions(candidate)
                for candidate in selected
            ]

            if any(
                not actions
                for actions in action_lists
            ):
                continue

            def expand(
                index: int,
                current: list,
            ):
                if index == len(selected):
                    scenarios.append(
                        tuple(current)
                    )
                    return

                candidate = selected[index]

                for action in action_lists[index]:
                    expand(
                        index + 1,
                        current + [
                            (candidate, action)
                        ],
                    )

            expand(0, [])

    # Deduplicate equivalent scenarios.
    unique = {}

    for scenario in scenarios:
        key = tuple(
            (
                candidate.series_id,
                action.value,
            )
            for candidate, action in scenario
        )

        unique[key] = tuple(scenario)

    return list(
        sorted(
            unique.values(),
            key=lambda scenario: (
                len(scenario),
                tuple(
                    (
                        candidate.series_id,
                        action.value,
                    )
                    for candidate, action in scenario
                ),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def decide(
    req: Request,
    profile: UserProfile,
    flows_base: list[CashFlow],
    options: list[PaymentOption],
    spending_candidates: list[SpendingChangeCandidate],
    horizon_start: date,
    horizon_end: date,
) -> tuple[
    Decision,
    Decimal,
    Optional[date],
]:
    """
    Produce the financial decision for one request.

    Architecture:

        BASELINE
            ↓
        amount_safe_to_pay
        earliest_date_for_full_payment
            ↓
        baseline plans
            ↓
        intervention plans
            ↓
        ranking
            ↓
        final Decision

    Baseline metrics are NEVER recomputed using spending changes.
    """

    starting_balance = (
        profile.current_available_balance
    )

    minimum = (
        profile.minimum_balance_to_keep
    )

    requested_amount = (
        req.requested_amount
    )

    # ---------------------------------------------------------------
    # 1. BASELINE METRICS
    # ---------------------------------------------------------------

    base_checkpoints = build_checkpoints(
        starting_balance,
        flows_base,
        horizon_end,
    )

    amount_safe_to_pay = (
        compute_amount_safe_to_pay(
            starting_balance,
            minimum,
            base_checkpoints,
            horizon_start,
            horizon_end,
            requested_amount,
        )
    )

    earliest_full_payment_date = (
        compute_earliest_full_payment_date(
            starting_balance,
            minimum,
            base_checkpoints,
            horizon_start,
            horizon_end,
            requested_amount,
        )
    )

    # ---------------------------------------------------------------
    # 2. BASELINE PLANS
    # ---------------------------------------------------------------

    baseline_plans = _eligible_plans(
        req,
        profile,
        flows_base,
        options,
        (),
        horizon_start,
        horizon_end,
    )

    partial = build_partial_candidate(
        req,
        profile,
        amount_safe_to_pay,
        earliest_full_payment_date,
    )

    if partial is not None:
        baseline_plans.append(partial)

    on_time = [
        candidate
        for candidate in baseline_plans
        if candidate.completes_by_deadline
    ]

    chosen: Optional[PlanCandidate] = None

    if on_time:
        chosen = min(
            on_time,
            key=rank_key,
        )

    # ---------------------------------------------------------------
    # 3. INTERVENTION PLANS
    # ---------------------------------------------------------------

    if chosen is None and spending_candidates:

        scenarios = _build_intervention_scenarios(
            spending_candidates,
            max_changes=4,
        )

        intervention_candidates: list[
            PlanCandidate
        ] = []

        for scenario in scenarios:

            if not scenario:
                continue

            changed_flows = (
                apply_spending_changes(
                    flows_base,
                    scenario,
                )
            )

            spending_changes = tuple(
                make_spending_change(
                    candidate,
                    action,
                )
                for candidate, action in scenario
            )

            plans = _eligible_plans(
                req,
                profile,
                changed_flows,
                options,
                spending_changes,
                horizon_start,
                horizon_end,
            )

            intervention_candidates.extend(
                candidate
                for candidate in plans
                if candidate.completes_by_deadline
            )

        if intervention_candidates:
            chosen = min(
                intervention_candidates,
                key=rank_key,
            )

    # ---------------------------------------------------------------
    # 4. WAIT / NOT AFFORDABLE
    # ---------------------------------------------------------------

    if chosen is None:

        methods = {
            method.lower()
            for method in profile.payment_methods_user_will_consider
        }

        if (
            earliest_full_payment_date is not None
            and "full_payment" in methods
        ):
            chosen = PlanCandidate(
                payment_method=PaymentMethod.WAIT,
                payments=[
                    PlanPayment(
                        earliest_full_payment_date,
                        requested_amount,
                    )
                ],
                spending_changes=[],
                completes_by_deadline=False,
                total_paid=requested_amount,
                first_payment_date=(
                    earliest_full_payment_date
                ),
                num_payments=1,
            )

            status = (
                AffordabilityStatus.AFFORDABLE_LATER
            )

        else:
            chosen = PlanCandidate(
                payment_method=(
                    PaymentMethod.NOT_RECOMMENDED
                ),
                payments=[],
                spending_changes=[],
                completes_by_deadline=False,
                total_paid=Decimal("0"),
                first_payment_date=req.request_date,
                num_payments=0,
            )

            status = (
                AffordabilityStatus.NOT_AFFORDABLE
            )

    # ---------------------------------------------------------------
    # 5. FINAL STATUS
    # ---------------------------------------------------------------

    else:

        if (
            chosen.payment_method
            == PaymentMethod.FULL_PAYMENT
        ):
            if chosen.spending_changes:
                status = (
                    AffordabilityStatus
                    .AFFORDABLE_WITH_PLAN
                )
            else:
                status = (
                    AffordabilityStatus
                    .AFFORDABLE_NOW
                )

        elif (
            chosen.payment_method
            == PaymentMethod.PARTIAL_PAYMENT
        ):
            status = (
                AffordabilityStatus
                .AFFORDABLE_WITH_PLAN
            )

        elif (
            chosen.payment_method
            == PaymentMethod.INSTALLMENTS
        ):
            status = (
                AffordabilityStatus
                .AFFORDABLE_WITH_PLAN
            )

        else:
            status = (
                AffordabilityStatus
                .AFFORDABLE_WITH_PLAN
            )
    # ---------------------------------------------------------------
    # 6. FINAL IMMUTABLE DECISION
    # ---------------------------------------------------------------

    if chosen.payment_method == PaymentMethod.WAIT:
        if earliest_full_payment_date is not None:
            explanation = (
                "The requested amount is not safely payable today while "
                "maintaining the required minimum balance. "
                f"Full payment is expected to become safe on "
                f"{earliest_full_payment_date.isoformat()}."
            )
        else:
            explanation = (
                "The requested amount cannot be safely paid while "
                "maintaining the required minimum balance."
            )

    elif chosen.payment_method == PaymentMethod.NOT_RECOMMENDED:
        explanation = (
            "The requested amount cannot be safely paid while "
            "maintaining the required minimum balance under the "
            "available payment options."
        )

    elif chosen.payment_method == PaymentMethod.FULL_PAYMENT:
        if chosen.spending_changes:
            changes = ", ".join(
                change.event_id
                for change in chosen.spending_changes
            )
            explanation = (
                "The payment can be made in full after applying the "
                f"allowed spending changes: {changes}. "
                f"The baseline safe-to-pay amount is "
                f"{amount_safe_to_pay:.2f}."
            )
        else:
            explanation = (
                "The requested amount can be paid in full while "
                "maintaining the required minimum balance."
            )

    elif chosen.payment_method == PaymentMethod.INSTALLMENTS:
        explanation = (
            "Pay using the selected installment plan. "
            "The scheduled payments remain within the available "
            "cash-flow constraints while maintaining the required "
            "minimum balance."
        )

    elif chosen.payment_method == PaymentMethod.PARTIAL_PAYMENT:
        explanation = (
            "Make the safe partial payment now and pay the remaining "
            "balance on the earliest safe date."
        )

    else:
        explanation = (
            "The selected payment plan satisfies the available "
            "cash-flow constraints."
        )

    decision = Decision(
        request_id=req.request_id,
        amount_safe_to_pay=amount_safe_to_pay,
        affordability_status=status,
        recommended_payment_method=chosen.payment_method,
        payment_plan=tuple(chosen.payments),
        earliest_date_for_full_payment=earliest_full_payment_date,
        spending_changes_needed=tuple(chosen.spending_changes),
        decision_explanation=explanation,
    )

    return (
        decision,
        amount_safe_to_pay,
        earliest_full_payment_date,
    )

# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def plan_request(
    req: Request,
    profile: UserProfile,
    facts: list,
    series_map: dict,
    options: list[PaymentOption],
    horizon_start: date,
    horizon_end: date,
) -> tuple[
    Decision,
    Decimal,
    Optional[date],
]:
    """
    Main entry point.

    Forecasting stays inside forecast.py.
    Planning stays inside planner.py.
    """

    flows = build_flows(
        facts,
        series_map,
        horizon_start,
        horizon_end,
    )

    spending_candidates = (
        build_spending_candidates(
            series_map,
            profile,
            horizon_start,
            horizon_end,
        )
    )

    return decide(
        req=req,
        profile=profile,
        flows_base=flows,
        options=options,
        spending_candidates=spending_candidates,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
    )
