from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path
from decimal import Decimal

from loader import load_dataset
from fx import FxConverter
from evidence import EvidenceResolver
from canonicalize import canonicalize_user
from series import build_series
from planner import plan_request
from usage import UsageTracker


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_PATH = ROOT / "output.csv"


HORIZON_DAYS = 90


def money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01")), "f")


def payment_plan_text(decision) -> str:
    if not decision.payment_plan:
        return "none"
    return "|".join(
        f"{payment.payment_date.isoformat()}:{money(payment.amount)}"
        for payment in decision.payment_plan
    )


def spending_changes_text(decision) -> str:
    parts = []

    for change in decision.spending_changes_needed:
        action = change.action.value

        if change.minimum_allowed_amount is not None:
            parts.append(
                f"{action}:{change.event_id}:{money(change.minimum_allowed_amount)}"
            )
        else:
            parts.append(
                f"{action}:{change.event_id}"
            )

    return "|".join(parts) if parts else "none"


def main() -> None:
    dataset = load_dataset(DATASET_DIR)

    fx = FxConverter(dataset.exchange_rates)

    usage = UsageTracker()

    resolver = EvidenceResolver(
    model="gemini-3.6-flash",
    usage_tracker=usage,
)

    rows = []

    for request_id in sorted(dataset.requests):
        request = dataset.requests[request_id]
        profile = dataset.profiles[request.user_id]

        events = dataset.events_by_user.get(
            request.user_id,
            [],
        )

        events_by_id = {
            event.event_id: event
            for event in events
        }

        # ---------------------------------------------------------------
        # Evidence resolution
        # ---------------------------------------------------------------

        messages = dataset.messages_by_user.get(
            request.user_id,
            [],
        )

        message_relations = resolver.resolve_messages(
            request.user_id,
            messages,
            events_by_id,
        )

        image_relations = []

        images = dataset.images_by_request.get(
            request.request_id,
            [],
        )

        for image in images:
            event = events_by_id.get(
                image.related_event_id
            )

            if event is None:
                continue

            relation = resolver.resolve_image(
                image,
                event,
            )

            if relation is not None:
                image_relations.append(relation)

        # ---------------------------------------------------------------
        # Canonical financial state
        # ---------------------------------------------------------------

        facts, series_overrides, series_terminations = (
            canonicalize_user(
                events=events,
                message_relations=message_relations,
                image_relations=image_relations,
                fx=fx,
                home_currency=profile.home_currency,
            )
        )

        # ---------------------------------------------------------------
        # Recurring state reconstruction
        # ---------------------------------------------------------------

        series_map = build_series(
            user_id=request.user_id,
            facts=facts,
            series_overrides=series_overrides,
            series_terminations=series_terminations,
            home_currency=profile.home_currency,
        )

        # ---------------------------------------------------------------
        # Forecast horizon
        #
        # Per problem_statement.md: "Forecast the user's balance for the
        # next 90 days." earliest_date_for_full_payment must be left empty
        # when the full amount isn't expected to become safe within THAT
        # window - not an arbitrarily long one. A wider window here made
        # almost everything "eventually" safe and answered a different
        # question than the one being asked.
        # ---------------------------------------------------------------

        horizon_start = request.request_date
        horizon_end = horizon_start + timedelta(days=HORIZON_DAYS)

        options = dataset.payment_options_by_request.get(
            request.request_id,
            [],
        )

        decision, _, _ = plan_request(
            req=request,
            profile=profile,
            facts=facts,
            series_map=series_map,
            options=options,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
        )

        rows.append(
            {
                "request_id": decision.request_id,
                "amount_safe_to_pay": money(
                    decision.amount_safe_to_pay
                ),
                "affordability_status": (
                    decision.affordability_status.value
                ),
                "recommended_payment_method": (
                    decision.recommended_payment_method.value
                ),
                "payment_plan": payment_plan_text(
                    decision
                ),
                "earliest_date_for_full_payment": (
                    decision.earliest_date_for_full_payment.isoformat()
                    if decision.earliest_date_for_full_payment
                    else ""
                ),
                "spending_changes_needed": (
                    spending_changes_text(decision)
                ),
                "decision_explanation": (
                    decision.decision_explanation
                ),
            }
        )

    fieldnames = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} decisions to {OUTPUT_PATH}"
    )

    usage_report_path = ROOT / "code" / "evaluation" / "usage_report.md"
    usage.save(str(usage_report_path), num_requests=len(rows))
    print(f"Wrote usage report to {usage_report_path}")


if __name__ == "__main__":
    main()