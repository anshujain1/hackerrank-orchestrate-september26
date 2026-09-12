from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

from models import (
    Direction,
    EventStatus,
    FinancialEvent,
    Flexibility,
    ImageEvidence,
    Message,
    PaymentOption,
    Request,
    UserProfile,
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _decimal(value: Optional[str]) -> Optional[Decimal]:
    """Parse an optional decimal without silently hiding malformed data."""
    if value is None or not value.strip():
        return None

    try:
        return Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal value: {value!r}") from exc


def _required_decimal(value: str, field_name: str) -> Decimal:
    """Parse a required monetary value."""
    result = _decimal(value)

    if result is None:
        raise ValueError(f"Missing required decimal field: {field_name}")

    return result


def _date(value: Optional[str]) -> Optional[date]:
    """Parse an optional ISO date."""
    if value is None or not value.strip():
        return None

    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError as exc:
        raise ValueError(f"Invalid date value: {value!r}") from exc


def _datetime(value: Optional[str]) -> datetime:
    """Parse an ISO datetime."""
    if value is None or not value.strip():
        raise ValueError("Missing required datetime value")

    value = value.strip()

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # Defensive fallback for timestamps without ISO formatting.
        try:
            return datetime.strptime(
                value[:19],
                "%Y-%m-%d %H:%M:%S",
            )
        except ValueError as exc:
            raise ValueError(f"Invalid datetime value: {value!r}") from exc


def _pipe(value: Optional[str]) -> tuple[str, ...]:
    """Convert pipe-separated CSV fields into immutable tuples."""
    if value is None or not value.strip():
        return ()

    return tuple(
        item.strip()
        for item in value.split("|")
        if item.strip()
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV using UTF-8 with BOM support."""
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    profiles: dict[str, UserProfile] = field(default_factory=dict)

    # Indexed by user because forecasting is user-centric.
    events_by_user: dict[str, list[FinancialEvent]] = field(
        default_factory=dict
    )

    # Requests are naturally addressed by request_id.
    requests: dict[str, Request] = field(default_factory=dict)

    # Payment options are naturally addressed by request_id.
    payment_options_by_request: dict[str, list[PaymentOption]] = field(
        default_factory=dict
    )

    # Evidence is commonly retrieved for a user.
    messages_by_user: dict[str, list[Message]] = field(
        default_factory=dict
    )

    # Images are commonly retrieved for a request.
    images_by_request: dict[str, list[ImageEvidence]] = field(
        default_factory=dict
    )

    # Fixed exchange-rate table supplied by the challenge.
    # (date, from_currency, to_currency, rate)
    exchange_rates: list[tuple[date, str, str, Decimal]] = field(
        default_factory=list
    )

    media_dir: Path = Path()


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def load_profiles(dataset_dir: Path) -> dict[str, UserProfile]:
    rows = _read_csv(dataset_dir / "financial_profiles.csv")

    profiles: dict[str, UserProfile] = {}

    for row in rows:
        user_id = row["user_id"].strip()

        profiles[user_id] = UserProfile(
            user_id=user_id,
            home_currency=row["home_currency"].strip(),

            current_available_balance=_required_decimal(
                row["current_available_balance"],
                "current_available_balance",
            ),

            minimum_balance_to_keep=_required_decimal(
                row["minimum_balance_to_keep"],
                "minimum_balance_to_keep",
            ),

            financial_priorities=_pipe(
                row["financial_priorities"]
            ),

            expense_categories_to_protect=_pipe(
                row["expense_categories_to_protect"]
            ),

            expense_categories_user_is_willing_to_reduce=_pipe(
                row["expense_categories_user_is_willing_to_reduce"]
            ),

            expense_categories_user_is_willing_to_stop=_pipe(
                row["expense_categories_user_is_willing_to_stop"]
            ),

            payment_methods_user_will_consider=_pipe(
                row["payment_methods_user_will_consider"]
            ),

            max_installment_months=(
                int(row["max_installment_months"].strip())
                if row["max_installment_months"].strip()
                else None
            ),
        )

    return profiles


# ---------------------------------------------------------------------------
# Financial events
# ---------------------------------------------------------------------------

def load_events(
    dataset_dir: Path,
) -> dict[str, list[FinancialEvent]]:
    rows = _read_csv(dataset_dir / "financial_events.csv")

    events_by_user: dict[str, list[FinancialEvent]] = {}

    for row in rows:
        event = FinancialEvent(
            event_id=row["event_id"].strip(),
            user_id=row["user_id"].strip(),

            event_type=row["event_type"].strip(),
            description=row["description"].strip(),
            category=row["category"].strip(),

            direction=Direction(
                row["direction"].strip().lower()
            ),

            amount=_decimal(row.get("amount")),

            currency=row["currency"].strip(),

            event_date=_date(row["event_date"]),
            settlement_date=_date(row.get("settlement_date")),

            status=EventStatus(
                row["status"].strip().lower()
            ),

            linked_event_id=(
                row["linked_event_id"].strip()
                if row.get("linked_event_id", "").strip()
                else None
            ),

            flexibility=Flexibility(
                row["flexibility"].strip().lower()
            ),

            minimum_allowed_amount=_decimal(
                row.get("minimum_allowed_amount")
            ),
        )

        # event_date is required by our model, so validate it explicitly.
        if event.event_date is None:
            raise ValueError(
                f"Missing event_date for {event.event_id}"
            )

        events_by_user.setdefault(
            event.user_id,
            [],
        ).append(event)

    return events_by_user


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

def load_requests(
    dataset_dir: Path,
) -> dict[str, Request]:
    rows = _read_csv(dataset_dir / "requests.csv")

    requests: dict[str, Request] = {}

    for row in rows:
        request_date = _date(row["request_date"])
        desired_completion_date = _date(
            row["desired_completion_date"]
        )

        if request_date is None:
            raise ValueError(
                f"Missing request_date for {row['request_id']}"
            )

        if desired_completion_date is None:
            raise ValueError(
                f"Missing desired_completion_date for "
                f"{row['request_id']}"
            )

        requests[row["request_id"].strip()] = Request(
            request_id=row["request_id"].strip(),
            user_id=row["user_id"].strip(),
            request_date=request_date,
            request_type=row["request_type"].strip(),

            requested_amount=_required_decimal(
                row["requested_amount"],
                "requested_amount",
            ),

            desired_completion_date=desired_completion_date,

            allows_partial_payment=(
                row["allows_partial_payment"]
                .strip()
                .lower()
                == "true"
            ),

            request_text=row["request_text"].strip(),
        )

    return requests


# ---------------------------------------------------------------------------
# Payment options
# ---------------------------------------------------------------------------

def load_payment_options(
    dataset_dir: Path,
) -> dict[str, list[PaymentOption]]:
    rows = _read_csv(
        dataset_dir / "request_payment_options.csv"
    )

    options_by_request: dict[str, list[PaymentOption]] = {}

    for row in rows:
        first_payment_date = _date(
            row["first_payment_date"]
        )

        if first_payment_date is None:
            raise ValueError(
                f"Missing first_payment_date for "
                f"{row['payment_option_id']}"
            )

        financing_fee = _decimal(
            row.get("financing_fee")
        )

        total_payable_amount = _decimal(
            row.get("total_payable_amount")
        )

        if financing_fee is None:
            financing_fee = Decimal("0")

        if total_payable_amount is None:
            raise ValueError(
                f"Missing total_payable_amount for "
                f"{row['payment_option_id']}"
            )

        option = PaymentOption(
            payment_option_id=row["payment_option_id"].strip(),
            request_id=row["request_id"].strip(),

            payment_method=row["payment_method"].strip(),

            payment_amount=_required_decimal(
                row["payment_amount"],
                "payment_amount",
            ),

            number_of_payments=int(
                row["number_of_payments"].strip()
            ),

            first_payment_date=first_payment_date,

            payment_frequency_days=(
                int(row["payment_frequency_days"].strip())
                if row.get("payment_frequency_days", "").strip()
                else None
            ),

            financing_fee=financing_fee,
            total_payable_amount=total_payable_amount,
        )

        options_by_request.setdefault(
            option.request_id,
            [],
        ).append(option)

    return options_by_request


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def load_messages(
    dataset_dir: Path,
) -> dict[str, list[Message]]:
    rows = _read_csv(dataset_dir / "messages.csv")

    messages_by_user: dict[str, list[Message]] = {}

    for row in rows:
        message = Message(
            message_id=row["message_id"].strip(),
            user_id=row["user_id"].strip(),

            request_id=(
                row["request_id"].strip()
                if row.get("request_id", "").strip()
                else None
            ),

            related_event_id=(
                row["related_event_id"].strip()
                if row.get("related_event_id", "").strip()
                else None
            ),

            sent_at=_datetime(row["sent_at"]),

            source_type=row["source_type"].strip(),
            message_text=row["message_text"].strip(),
        )

        messages_by_user.setdefault(
            message.user_id,
            [],
        ).append(message)

    return messages_by_user


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def load_images(
    dataset_dir: Path,
) -> dict[str, list[ImageEvidence]]:
    rows = _read_csv(dataset_dir / "images.csv")

    images_by_request: dict[str, list[ImageEvidence]] = {}

    for row in rows:
        image_id = row["image_id"].strip()

        image = ImageEvidence(
            image_id=image_id,
            user_id=row["user_id"].strip(),

            request_id=(
                row["request_id"].strip()
                if row.get("request_id", "").strip()
                else None
            ),

            related_event_id=(
                row["related_event_id"].strip()
                if row.get("related_event_id", "").strip()
                else None
            ),

            path=(
                dataset_dir
                / "media"
                / "images"
                / f"{image_id}.png"
            ),
        )

        if image.request_id:
            images_by_request.setdefault(
                image.request_id,
                [],
            ).append(image)

    return images_by_request


# ---------------------------------------------------------------------------
# Exchange rates
# ---------------------------------------------------------------------------

def load_exchange_rates(
    dataset_dir: Path,
) -> list[tuple[date, str, str, Decimal]]:
    rows = _read_csv(dataset_dir / "exchange_rates.csv")

    rates: list[tuple[date, str, str, Decimal]] = []

    for row in rows:
        rate_date = _date(row["rate_date"])

        if rate_date is None:
            raise ValueError(
                f"Missing rate_date in exchange_rates.csv"
            )

        rate = _required_decimal(
            row["rate"],
            "rate",
        )

        rates.append(
            (
                rate_date,
                row["from_currency"].strip(),
                row["to_currency"].strip(),
                rate,
            )
        )

    return rates


# ---------------------------------------------------------------------------
# Complete dataset loader
# ---------------------------------------------------------------------------

def load_dataset(dataset_dir: str | Path) -> Dataset:
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    return Dataset(
        profiles=load_profiles(dataset_dir),
        events_by_user=load_events(dataset_dir),
        requests=load_requests(dataset_dir),
        payment_options_by_request=load_payment_options(
            dataset_dir
        ),
        messages_by_user=load_messages(dataset_dir),
        images_by_request=load_images(dataset_dir),
        exchange_rates=load_exchange_rates(dataset_dir),
        media_dir=dataset_dir / "media" / "images",
    )