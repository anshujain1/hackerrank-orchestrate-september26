from __future__ import annotations

import csv
from datetime import date, datetime
from dataclasses import dataclass
from decimal import Decimal
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


def _decimal(value: str) -> Optional[Decimal]:
    value = value.strip()
    if not value:
        return None
    return Decimal(value)


def _date(value: str) -> date:
    return date.fromisoformat(value.strip())


def _datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def _tuple_field(value: str) -> tuple[str, ...]:
    value = value.strip()
    if not value:
        return ()
    return tuple(part.strip() for part in value.split("|") if part.strip())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_profiles(dataset_dir: Path) -> dict[str, UserProfile]:
    rows = _read_csv(dataset_dir / "financial_profiles.csv")

    profiles: dict[str, UserProfile] = {}

    for row in rows:
        user_id = row["user_id"]

        profiles[user_id] = UserProfile(
            user_id=user_id,
            home_currency=row["home_currency"].strip(),
            current_available_balance=Decimal(
                row["current_available_balance"].strip()
            ),
            minimum_balance_to_keep=Decimal(
                row["minimum_balance_to_keep"].strip()
            ),
            financial_priorities=_tuple_field(row["financial_priorities"]),
            expense_categories_to_protect=_tuple_field(
                row["expense_categories_to_protect"]
            ),
            expense_categories_user_is_willing_to_reduce=_tuple_field(
                row["expense_categories_user_is_willing_to_reduce"]
            ),
            expense_categories_user_is_willing_to_stop=_tuple_field(
                row["expense_categories_user_is_willing_to_stop"]
            ),
            payment_methods_user_will_consider=_tuple_field(
                row["payment_methods_user_will_consider"]
            ),
            max_installment_months=(
                int(row["max_installment_months"].strip())
                if row["max_installment_months"].strip()
                else None
            ),
        )

    return profiles


def load_events(dataset_dir: Path) -> list[FinancialEvent]:
    rows = _read_csv(dataset_dir / "financial_events.csv")

    events: list[FinancialEvent] = []

    for row in rows:
        events.append(
            FinancialEvent(
                event_id=row["event_id"],
                user_id=row["user_id"],
                event_type=row["event_type"].strip(),
                description=row["description"].strip(),
                category=row["category"].strip(),
                direction=Direction(row["direction"].strip().lower()),
                amount=_decimal(row["amount"]),
                currency=row["currency"].strip(),
                event_date=_date(row["event_date"]),
                settlement_date=(
                    _date(row["settlement_date"])
                    if row["settlement_date"].strip()
                    else None
                ),
                status=EventStatus(row["status"].strip().lower()),
                linked_event_id=(
                    row["linked_event_id"].strip()
                    if row["linked_event_id"].strip()
                    else None
                ),
                flexibility=Flexibility(row["flexibility"].strip().lower()),
                minimum_allowed_amount=_decimal(
                    row["minimum_allowed_amount"]
                ),
            )
        )

    return events


def load_requests(dataset_dir: Path) -> list[Request]:
    rows = _read_csv(dataset_dir / "requests.csv")

    requests: list[Request] = []

    for row in rows:
        requests.append(
            Request(
                request_id=row["request_id"],
                user_id=row["user_id"],
                request_date=_date(row["request_date"]),
                request_type=row["request_type"].strip(),
                requested_amount=Decimal(row["requested_amount"].strip()),
                desired_completion_date=_date(
                    row["desired_completion_date"]
                ),
                allows_partial_payment=(
                    row["allows_partial_payment"].strip().lower()
                    == "true"
                ),
                request_text=row["request_text"].strip(),
            )
        )

    return requests


def load_payment_options(dataset_dir: Path) -> list[PaymentOption]:
    rows = _read_csv(dataset_dir / "request_payment_options.csv")

    options: list[PaymentOption] = []

    for row in rows:
        options.append(
            PaymentOption(
                payment_option_id=row["payment_option_id"],
                request_id=row["request_id"],
                payment_method=row["payment_method"].strip(),
                payment_amount=Decimal(row["payment_amount"].strip()),
                number_of_payments=int(row["number_of_payments"].strip()),
                first_payment_date=_date(row["first_payment_date"]),
                payment_frequency_days=(
                    int(row["payment_frequency_days"].strip())
                    if row["payment_frequency_days"].strip()
                    else None
                ),
                financing_fee=Decimal(row["financing_fee"].strip()),
                total_payable_amount=Decimal(
                    row["total_payable_amount"].strip()
                ),
            )
        )

    return options


def load_messages(dataset_dir: Path) -> list[Message]:
    rows = _read_csv(dataset_dir / "messages.csv")

    messages: list[Message] = []

    for row in rows:
        messages.append(
            Message(
                message_id=row["message_id"],
                user_id=row["user_id"],
                request_id=(
                    row["request_id"].strip()
                    if row["request_id"].strip()
                    else None
                ),
                related_event_id=(
                    row["related_event_id"].strip()
                    if row["related_event_id"].strip()
                    else None
                ),
                sent_at=_datetime(row["sent_at"]),
                source_type=row["source_type"].strip(),
                message_text=row["message_text"].strip(),
            )
        )

    return messages


def load_images(dataset_dir: Path) -> list[ImageEvidence]:
    rows = _read_csv(dataset_dir / "images.csv")

    images: list[ImageEvidence] = []

    for row in rows:
        image_id = row["image_id"]

        images.append(
            ImageEvidence(
                image_id=image_id,
                user_id=row["user_id"],
                request_id=(
                    row["request_id"].strip()
                    if row["request_id"].strip()
                    else None
                ),
                related_event_id=(
                    row["related_event_id"].strip()
                    if row["related_event_id"].strip()
                    else None
                ),
                path=dataset_dir / "media" / "images" / f"{image_id}.png",
            )
        )

    return images


@dataclass
class Dataset:
    profiles: dict[str, UserProfile]
    events: list[FinancialEvent]
    requests: list[Request]
    payment_options: list[PaymentOption]
    messages: list[Message]
    images: list[ImageEvidence]


def load_dataset(dataset_dir: Path) -> Dataset:
    return Dataset(
        profiles=load_profiles(dataset_dir),
        events=load_events(dataset_dir),
        requests=load_requests(dataset_dir),
        payment_options=load_payment_options(dataset_dir),
        messages=load_messages(dataset_dir),
        images=load_images(dataset_dir),
    )