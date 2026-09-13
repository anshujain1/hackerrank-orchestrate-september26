from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Optional

from models import (
    AmountOverride,
    Direction,
    EventStatus,
    EvidenceRelation,
    EvidenceRelationType,
    FinancialEvent,
    FinancialFact,
    FinancialRole,
)


# ---------------------------------------------------------------------------
# Category knowledge
# ---------------------------------------------------------------------------

KNOWN_RECURRING_CATEGORIES = {
    "salary",
    "rent",
    "housing",
    "utilities",
    "education",
    "debt_repayment",
    "music_subscription",
    "delivery_membership",
    "insurance",
    "healthcare",
    "family_support",
    "work_expense",
    "gym",
    "streaming",
    "cloud_storage",
    "groceries",
    "transport",
    "dining",
    "entertainment",
}

TRANSFER_CATEGORIES = {
    "family_transfer",
}


# Evidence below this level is not allowed to mutate canonical state.
CONFIDENCE_THRESHOLD = Decimal("0.55")


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

def classify_role(event: FinancialEvent) -> FinancialRole:
    """
    Assign the financial role of a raw event.

    Semantic interpretation of messages/images happens in evidence.py.
    This function only interprets structured event fields.
    """

    if event.event_type == "refund":
        return FinancialRole.REFUND

    if (
        event.event_type == "investment_valuation"
        or event.status == EventStatus.UNREALIZED
        or event.direction == Direction.NON_CASH
    ):
        return FinancialRole.NON_CASH

    if event.category in TRANSFER_CATEGORIES:
        return FinancialRole.TRANSFER

    if event.category in KNOWN_RECURRING_CATEGORIES:
        if event.direction == Direction.CREDIT:
            return FinancialRole.RECURRING_INCOME

        if event.direction == Direction.DEBIT:
            return FinancialRole.RECURRING_EXPENSE

    if event.direction == Direction.CREDIT:
        return FinancialRole.ONE_TIME_INCOME

    if event.direction == Direction.DEBIT:
        return FinancialRole.ONE_TIME_EXPENSE

    return FinancialRole.UNKNOWN


# ---------------------------------------------------------------------------
# Forecast eligibility
# ---------------------------------------------------------------------------

def counts_in_forecast(
    event: FinancialEvent,
    forced_uncertain: bool = False,
) -> bool:
    """
    Decide whether a concrete event is eligible to affect the forecast.

    Rules:
    - cancelled/failed/unrealized events do not count
    - non-cash events do not count
    - pending credits do not count
    - uncertain future events do not count
    """

    if event.status in {
        EventStatus.CANCELLED,
        EventStatus.FAILED,
        EventStatus.UNREALIZED,
    }:
        return False

    if event.direction == Direction.NON_CASH:
        return False

    if (
        event.status == EventStatus.PENDING
        and event.direction == Direction.CREDIT
    ):
        return False

    if forced_uncertain:
        return False

    return True


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def _trusted(
    relations: list[EvidenceRelation],
) -> list[EvidenceRelation]:
    """Keep only sufficiently confident evidence relations."""

    return [
        relation
        for relation in relations
        if relation.confidence >= CONFIDENCE_THRESHOLD
    ]


def _latest_relation(
    relations: list[EvidenceRelation],
    relation_type: EvidenceRelationType,
) -> Optional[EvidenceRelation]:
    """
    Return the strongest/latest usable relation of a particular type.

    Confidence is the primary ordering criterion. Original evidence order
    is retained as a deterministic tie-breaker.
    """

    candidates = [
        relation
        for relation in relations
        if relation.relation == relation_type
    ]

    if not candidates:
        return None

    return max(
        enumerate(candidates),
        key=lambda item: (item[1].confidence, item[0]),
    )[1]


# ---------------------------------------------------------------------------
# Main canonicalization
# ---------------------------------------------------------------------------

def canonicalize_user(
    events: list[FinancialEvent],
    message_relations: list[EvidenceRelation],
    image_relations: list[EvidenceRelation],
    fx,
    home_currency: str,
) -> tuple[
    list[FinancialFact],
    list[tuple[str, AmountOverride]],
    list[tuple[str, object]],
]:
    """
    Convert raw FinancialEvent records into canonical FinancialFact records.

    Returns
    -------
    facts:
        Canonical event-level financial facts.

    series_overrides:
        Tuples of (event_id, AmountOverride) so series.py can determine
        which recurring series the amendment belongs to.

    series_terminations:
        Tuples of (event_id, termination_date).

    Design principles
    -----------------
    1. Evidence modifies state; it does not make affordability decisions.
    2. Blank amounts remain None until image evidence resolves them.
    3. Duplicate suppression requires explicit duplicate evidence.
    4. Internal transfers remain in history.
    5. Forecast eligibility is explicit.
    6. Evidence provenance is preserved.
    """

    # ------------------------------------------------------------------
    # Combine and filter evidence
    # ------------------------------------------------------------------

    all_relations = _trusted(
        list(message_relations) + list(image_relations)
    )

    relation_by_event: dict[
        str,
        list[EvidenceRelation],
    ] = {}

    for relation in all_relations:
        if relation.target_id:
            relation_by_event.setdefault(
                relation.target_id,
                [],
            ).append(relation)

    # ------------------------------------------------------------------
    # Explicit duplicate suppression
    #
    # We NEVER deduplicate merely because rows look similar.
    # Similarity is evidence for investigation, not proof.
    # ------------------------------------------------------------------

    duplicate_event_ids: set[str] = set()

    for relation in all_relations:
        if relation.relation != EvidenceRelationType.DUPLICATE:
            continue

        if relation.target_id:
            duplicate_event_ids.add(relation.target_id)

    # ------------------------------------------------------------------
    # Raw event lookup
    # ------------------------------------------------------------------

    events_by_id = {
        event.event_id: event
        for event in events
    }

    # Keep these in the exact format expected by series.py.
    series_overrides: list[
        tuple[str, AmountOverride]
    ] = []

    series_terminations: list[
        tuple[str, object]
    ] = []

    facts: list[FinancialFact] = []

    # ------------------------------------------------------------------
    # Process events chronologically
    # ------------------------------------------------------------------

    for event in sorted(
        events,
        key=lambda item: (
            item.event_date,
            item.event_id,
        ),
    ):

        # --------------------------------------------------------------
        # Explicit duplicate
        # --------------------------------------------------------------

        if event.event_id in duplicate_event_ids:
            continue

        event_relations = relation_by_event.get(
            event.event_id,
            [],
        )

        # --------------------------------------------------------------
        # Cancellation
        # --------------------------------------------------------------

        cancellation = _latest_relation(
            event_relations,
            EvidenceRelationType.CANCELLATION,
        )

        if cancellation is not None:
            event = replace(
                event,
                status=EventStatus.CANCELLED,
            )

        # --------------------------------------------------------------
        # Settlement
        # --------------------------------------------------------------

        settlement = _latest_relation(
            event_relations,
            EvidenceRelationType.SETTLEMENT,
        )

        if settlement is not None:
            event = replace(
                event,
                status=EventStatus.SETTLED,
            )

        # --------------------------------------------------------------
        # Delay
        #
        # IMPORTANT:
        # FinancialEvent uses event_date.
        # FinancialFact uses date.
        # --------------------------------------------------------------

        delay = _latest_relation(
            event_relations,
            EvidenceRelationType.DELAY,
        )

        if (
            delay is not None
            and delay.new_date is not None
        ):
            event = replace(
                event,
                event_date=delay.new_date,
            )

        # --------------------------------------------------------------
        # Explicit amendment
        # --------------------------------------------------------------

        amendment = _latest_relation(
            event_relations,
            EvidenceRelationType.AMENDMENT,
        )

        if amendment is not None:

            effective_date = (
                amendment.effective_date
                or amendment.new_date
                or event.event_date
            )

            # ----------------------------------------------------------
            # Amendment to this concrete occurrence
            # ----------------------------------------------------------

            if (
                amendment.new_amount is not None
                and effective_date == event.event_date
            ):
                event = replace(
                    event,
                    amount=amendment.new_amount,
                    currency=(
                        amendment.new_currency
                        or event.currency
                    ),
                )

            if (
                amendment.new_date is not None
                and effective_date == event.event_date
            ):
                event = replace(
                    event,
                    event_date=amendment.new_date,
                )

            # ----------------------------------------------------------
            # Future recurring-series amount change
            # ----------------------------------------------------------

            elif (
                amendment.new_amount is not None
                and effective_date > event.event_date
            ):
                series_overrides.append(
                    (
                        event.event_id,
                        AmountOverride(
                            effective_date=effective_date,
                            amount=amendment.new_amount,
                            currency=(
                                amendment.new_currency
                                or event.currency
                            ),
                            evidence_ids=amendment.evidence_ids,
                        ),
                    )
                )

        # --------------------------------------------------------------
        # Uncertainty
        # --------------------------------------------------------------

        uncertainty = _latest_relation(
            event_relations,
            EvidenceRelationType.UNCERTAINTY,
        )

        forced_uncertain = uncertainty is not None

        # --------------------------------------------------------------
        # Termination
        # --------------------------------------------------------------

        termination = _latest_relation(
            event_relations,
            EvidenceRelationType.TERMINATION,
        )

        if termination is not None:

            effective_date = (
                termination.effective_date
                or termination.new_date
            )

            if effective_date is not None:
                series_terminations.append(
                    (
                        event.event_id,
                        effective_date,
                    )
                )

        # --------------------------------------------------------------
        # Amount
        #
        # CRITICAL:
        # Blank amount is NOT zero.
        # --------------------------------------------------------------

        amount = event.amount
        currency = event.currency

        home_amount = None

        if amount is not None:
            home_amount = fx.convert(
                amount,
                currency,
                home_currency,
                event.event_date,
            )

        # --------------------------------------------------------------
        # Evidence provenance
        # --------------------------------------------------------------

        evidence_ids = tuple(
            sorted(
                {
                    evidence_id
                    for relation in event_relations
                    for evidence_id in relation.evidence_ids
                }
            )
        )

        # --------------------------------------------------------------
        # Financial role
        # --------------------------------------------------------------

        role = classify_role(event)

        # --------------------------------------------------------------
        # Forecast eligibility
        # --------------------------------------------------------------

        include = counts_in_forecast(
            event,
            forced_uncertain=forced_uncertain,
        )

        # A concrete event with unresolved amount cannot safely
        # participate in arithmetic.
        if amount is None:
            include = False

        # --------------------------------------------------------------
        # Canonical fact
        #
        # Raw event: event.event_date
        # Canonical fact: date
        # --------------------------------------------------------------

        facts.append(
    FinancialFact(
        event_id=event.event_id,
        user_id=event.user_id,
        date=event.event_date,
        settlement_date=event.settlement_date,
        amount=amount,
        currency=currency,
        home_currency_amount=home_amount,
        direction=event.direction,
        category=event.category,
        description=event.description,
        financial_role=role,
        status=event.status,
        counts_in_forecast=include,
        flexibility=event.flexibility,
        minimum_allowed_amount=event.minimum_allowed_amount,
        evidence_ids=evidence_ids,
    )
)

    return (
        facts,
        series_overrides,
        series_terminations,
    )