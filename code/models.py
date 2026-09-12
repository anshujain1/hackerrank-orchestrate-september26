from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Core financial enums
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"
    NON_CASH = "non_cash"


class EventStatus(str, Enum):
    SETTLED = "settled"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNREALIZED = "unrealized"


class Flexibility(str, Enum):
    FIXED = "fixed"
    REDUCIBLE = "reducible"
    STOPPABLE = "stoppable"
    REDUCIBLE_OR_STOPPABLE = "reducible_or_stoppable"


class FinancialRole(str, Enum):
    RECURRING_INCOME = "recurring_income"
    ONE_TIME_INCOME = "one_time_income"
    RECURRING_EXPENSE = "recurring_expense"
    ONE_TIME_EXPENSE = "one_time_expense"
    TRANSFER = "transfer"
    REFUND = "refund"
    NON_CASH = "non_cash"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Evidence resolution
# ---------------------------------------------------------------------------

class EvidenceRelationType(str, Enum):
    DUPLICATE = "duplicate"
    AMENDMENT = "amendment"
    CANCELLATION = "cancellation"
    SETTLEMENT = "settlement"
    DELAY = "delay"
    CONFIRMATION = "confirmation"
    UNCERTAINTY = "uncertainty"
    COMPONENT = "component"
    TERMINATION = "termination"

    # Important: TRANSFER does NOT mean "net these events out".
    # Internal transfers remain as separate ledger history unless the
    # financial rules explicitly say otherwise.
    TRANSFER = "transfer"

    UNRELATED = "unrelated"


# ---------------------------------------------------------------------------
# Output enums
# ---------------------------------------------------------------------------

class AffordabilityStatus(str, Enum):
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"


class PaymentMethod(str, Enum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class SpendingChangeAction(str, Enum):
    STOP = "stop"
    REDUCE_TO = "reduce_to"


# ---------------------------------------------------------------------------
# Raw input records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: Direction
    amount: Optional[Decimal]
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: EventStatus
    linked_event_id: Optional[str]
    flexibility: Flexibility
    minimum_allowed_amount: Optional[Decimal]


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[str, ...]
    max_installment_months: Optional[int]


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]
    financing_fee: Decimal
    total_payable_amount: Decimal


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: datetime
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageEvidence:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    path: Path


# ---------------------------------------------------------------------------
# Evidence relations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceRelation:
    """
    A relationship between evidence records.

    The optional payload fields allow an evidence item to introduce a new
    financial fact, rather than merely describing a relationship.

    Examples:
        salary raise:
            relation=AMENDMENT
            new_amount=Decimal("42750000")
            effective_date=...

        salary delay:
            relation=DELAY
            new_date=...

        seasonal contract ending:
            relation=TERMINATION
            effective_date=...
    """

    source_id: str
    target_id: Optional[str]
    relation: EvidenceRelationType
    confidence: Decimal

    new_amount: Optional[Decimal] = None
    new_currency: Optional[str] = None
    new_date: Optional[date] = None
    effective_date: Optional[date] = None

    evidence_ids: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Canonical financial layer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinancialFact:
    """
    A canonical financial fact derived from one or more raw events/evidence.

    `category` and `description` intentionally survive normalization because
    downstream forecasting and flexible-spending decisions need them.

    `home_currency_amount` is the normalized amount used by the forecast.
    """

    event_id: str
    user_id: str

    date: date
    settlement_date: Optional[date]

    amount: Decimal
    currency: str
    home_currency_amount: Decimal

    direction: Direction
    category: str
    description: str

    financial_role: FinancialRole
    status: EventStatus

    # Whether this fact should affect the deterministic forecast.
    counts_in_forecast: bool

    flexibility: Flexibility
    minimum_allowed_amount: Optional[Decimal]

    evidence_ids: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Recurring financial streams
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AmountOverride:
    """
    A change to a recurring series beginning on a specific date.

    Example:
        salary increases from 40M to 42.75M starting 2025-08-15.
    """

    effective_date: date
    amount: Decimal
    currency: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecurringSeries:
    """
    A recurring financial pattern inferred from multiple canonical facts.

    Recurrence is deliberately NOT stored on FinancialFact. It is inferred
    from the series as a whole.
    """

    series_id: str
    user_id: str
    category: str
    direction: Direction

    typical_amount: Decimal
    currency: str
    interval_days: int

    # Last real, deduplicated occurrence supporting the series.
    anchor_date: date

    flexibility: Flexibility
    minimum_allowed_amount: Optional[Decimal]

    member_fact_ids: tuple[str, ...]

    amount_overrides: tuple[AmountOverride, ...] = ()

    # If set, occurrences after this date must not be forecast.
    terminated_after: Optional[date] = None


# ---------------------------------------------------------------------------
# Spending changes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpendingChange:
    action: SpendingChangeAction
    event_id: str
    new_amount: Optional[Decimal] = None


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForecastPoint:
    on_date: date
    delta: Decimal
    balance_after: Decimal

    # The financial fact responsible for the movement, when applicable.
    fact_id: Optional[str]

    reason: str


@dataclass(frozen=True)
class ForecastResult:
    points: tuple[ForecastPoint, ...]

    min_balance: Decimal
    min_balance_date: Optional[date]

    breached: bool


# ---------------------------------------------------------------------------
# Payment planning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanPayment:
    payment_date: date
    amount: Decimal


@dataclass
class PlanCandidate:
    """
    A candidate plan before the final Decision is selected.

    Ranking is deliberately separate from Decision construction.
    """

    payment_method: PaymentMethod

    payments: list[PlanPayment]
    spending_changes: list[SpendingChange]

    completes_by_deadline: bool

    total_paid: Decimal
    first_payment_date: date
    num_payments: int

    payment_option_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Final decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    request_id: str

    amount_safe_to_pay: Decimal

    affordability_status: AffordabilityStatus
    recommended_payment_method: PaymentMethod

    payment_plan: tuple[PlanPayment, ...]

    earliest_date_for_full_payment: Optional[date]

    spending_changes_needed: tuple[SpendingChange, ...]

    decision_explanation: str

