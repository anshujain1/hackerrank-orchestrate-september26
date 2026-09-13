import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

from loader import load_dataset, Request
from fx import FxConverter
from evidence import EvidenceResolver
from canonicalize import canonicalize_user
from series import build_series
from planner import plan_request


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"

dataset = load_dataset(DATASET_DIR)
fx = FxConverter(dataset.exchange_rates)

sample_path = DATASET_DIR / "sample_requests.csv"
with sample_path.open(encoding="utf-8") as f:
    samples = list(csv.DictReader(f))

resolver = EvidenceResolver(model="gemini-3.6-flash")

fields = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]

def money(x):
    return f"{Decimal(x).quantize(Decimal('0.01')):.2f}"

def plan_text(decision):
    if not decision.payment_plan:
        return "none"
    return "|".join(
        f"{p.payment_date.isoformat()}:{money(p.amount)}"
        for p in decision.payment_plan
    )

def changes_text(decision):
    if not decision.spending_changes_needed:
        return "none"
    parts = []
    for c in decision.spending_changes_needed:
        if c.minimum_allowed_amount is not None:
            parts.append(f"{c.action.value}:{c.event_id}:{money(c.minimum_allowed_amount)}")
        else:
            parts.append(f"{c.action.value}:{c.event_id}")
    return "|".join(parts)

passed = 0

for row in samples:
    req = Request(
        request_id=row["request_id"],
        user_id=row["user_id"],
        request_date=date.fromisoformat(row["request_date"]),
        request_type=row["request_type"],
        requested_amount=Decimal(row["requested_amount"]),
        desired_completion_date=date.fromisoformat(row["desired_completion_date"]),
        allows_partial_payment=row["allows_partial_payment"].lower() == "true",
        request_text=row["request_text"],
    )

    profile = dataset.profiles[req.user_id]
    events = dataset.events_by_user.get(req.user_id, [])
    events_by_id = {e.event_id: e for e in events}

    messages = dataset.messages_by_user.get(req.user_id, [])
    message_relations = resolver.resolve_messages(
        req.user_id, messages, events_by_id
    )

    image_relations = []
    for image in dataset.images_by_request.get(req.request_id, []):
        event = events_by_id.get(image.related_event_id)
        if event is not None:
            relation = resolver.resolve_image(image, event)
            if relation is not None:
                image_relations.append(relation)

    facts, series_overrides, series_terminations = canonicalize_user(
        events=events,
        message_relations=message_relations,
        image_relations=image_relations,
        fx=fx,
        home_currency=profile.home_currency,
    )

    series_map = build_series(
        user_id=req.user_id,
        facts=facts,
        series_overrides=series_overrides,
        series_terminations=series_terminations,
        home_currency=profile.home_currency,
    )

    options = dataset.payment_options_by_request.get(req.request_id, [])

    decision, _, _ = plan_request(
        req=req,
        profile=profile,
        facts=facts,
        series_map=series_map,
        options=options,
        horizon_start=req.request_date,
        horizon_end=req.request_date.replace(day=req.request_date.day) + __import__("datetime").timedelta(days=90),
    )

    actual = {
        "amount_safe_to_pay": money(decision.amount_safe_to_pay),
        "affordability_status": decision.affordability_status.value,
        "recommended_payment_method": decision.recommended_payment_method.value,
        "payment_plan": plan_text(decision),
        "earliest_date_for_full_payment": (
            decision.earliest_date_for_full_payment.isoformat()
            if decision.earliest_date_for_full_payment else ""
        ),
        "spending_changes_needed": changes_text(decision),
    }

    mismatches = []
    for field in fields:
        expected = row[field]
        if field == "amount_safe_to_pay":
            expected = money(expected)
        if actual[field] != expected:
            mismatches.append(f"{field}: expected={expected!r}, actual={actual[field]!r}")

    if mismatches:
        print(f"FAIL {req.request_id}")
        for m in mismatches:
            print("  " + m)
    else:
        print(f"PASS {req.request_id}")
        passed += 1

print()
print(f"RESULT: {passed}/{len(samples)} passed")
