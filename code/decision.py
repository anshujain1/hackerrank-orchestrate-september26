from __future__ import annotations

from datetime import date
from decimal import Decimal

from models import (
    AffordabilityStatus,
    Decision,
    PaymentMethod,
    PlanCandidate,
)


def _status_for_candidate(candidate: PlanCandidate) -> AffordabilityStatus:
    """
    Convert the planner's winning payment candidate into the final
    affordability status.

    The planner has already determined whether the candidate is safe
    and whether it requires an intervention or delayed payment.
    """

    if candidate.payment_method == PaymentMethod.WAIT:
        return AffordabilityStatus.AFFORDABLE_LATER

    if candidate.payment_method == PaymentMethod.FULL_PAYMENT:
        if candidate.spending_changes:
            return AffordabilityStatus.AFFORDABLE_WITH_PLAN

        return AffordabilityStatus.AFFORDABLE_NOW

    if candidate.payment_method in (
        PaymentMethod.PARTIAL_PAYMENT,
        PaymentMethod.INSTALLMENTS,
    ):
        return AffordabilityStatus.AFFORDABLE_WITH_PLAN

    return AffordabilityStatus.NOT_AFFORDABLE


def _explanation(
    candidate: PlanCandidate,
    amount_safe_to_pay: Decimal,
    earliest_date_for_full_payment: date | None,
) -> str:
    """
    Produce a concise deterministic explanation.

    No LLM is needed here: the explanation is generated from the
    already-computed financial decision.
    """

    if candidate.payment_method == PaymentMethod.WAIT:
        if earliest_date_for_full_payment is not None:
            return (
                f"The requested amount is not safely payable today while "
                f"maintaining the required minimum balance. Full payment "
                f"is expected to become safe on "
                f"{earliest_date_for_full_payment.isoformat()}."
            )

        return (
            "The requested amount is not safely payable today while "
            "maintaining the required minimum balance."
        )

    if candidate.payment_method == PaymentMethod.FULL_PAYMENT:
        if candidate.spending_changes:
            changes = ", ".join(
                change.event_id for change in candidate.spending_changes
            )
            return (
                f"The payment can be made in full after applying the "
                f"allowed spending changes: {changes}. "
                f"The baseline safe-to-pay amount is "
                f"{amount_safe_to_pay}."
            )

        return (
            f"The requested amount can be paid in full while maintaining "
            f"the required minimum balance. The baseline safe-to-pay "
            f"amount is {amount_safe_to_pay}."
        )

    if candidate.payment_method == PaymentMethod.PARTIAL_PAYMENT:
        return (
            f"The full amount cannot be safely paid immediately, but a "
            f"partial payment is feasible under the request's terms. "
            f"The baseline safe-to-pay amount is {amount_safe_to_pay}."
        )

    if candidate.payment_method == PaymentMethod.INSTALLMENTS:
        return (
            f"The full amount is not safely payable immediately, but the "
            f"available installment plan fits the financial forecast."
        )

    return (
        "The requested payment is not safely recommended under the "
        "available financial constraints."
    )


def make_decision(
    request_id: str,
    candidate: PlanCandidate,
    amount_safe_to_pay: Decimal,
    earliest_date_for_full_payment: date | None,
) -> Decision:
    """
    Freeze the planner's winning candidate into the final Decision object.

    This function intentionally contains no financial forecasting logic.
    """

    status = _status_for_candidate(candidate)

    return Decision(
        request_id=request_id,
        amount_safe_to_pay=amount_safe_to_pay,
        affordability_status=status,
        recommended_payment_method=candidate.payment_method,
        payment_plan=tuple(candidate.payments),
        earliest_date_for_full_payment=earliest_date_for_full_payment,
        spending_changes_needed=tuple(candidate.spending_changes),
        decision_explanation=_explanation(
            candidate,
            amount_safe_to_pay,
            earliest_date_for_full_payment,
        ),
    )