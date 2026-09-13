from pathlib import Path

from loader import load_dataset
from fx import FxConverter
from canonicalize import canonicalize_user
from series import build_series

d = load_dataset(Path("..") / "dataset")
fx = FxConverter(d.exchange_rates)

for user_id in ["user_02", "user_06", "user_25"]:
    print("\n" + "=" * 80)
    print("USER:", user_id)

    events = d.events_by_user[user_id]

    facts, overrides, terminations = canonicalize_user(
        events=events,
        message_relations=[],
        image_relations=[],
        fx=fx,
        home_currency=d.profiles[user_id].home_currency,
    )

    series = build_series(
        user_id=user_id,
        facts=facts,
        series_overrides=overrides,
        series_terminations=terminations,
        home_currency=d.profiles[user_id].home_currency,
    )

    print("FACTS:", len(facts))
    print("SERIES:", len(series))

    for sid, s in series.items():
        print(
            sid,
            "| category=", s.category,
            "| direction=", s.direction,
            "| typical=", s.typical_amount,
            "| interval=", s.interval_days,
            "| anchor=", s.anchor_date,
            "| flexibility=", s.flexibility,
            "| min=", s.minimum_allowed_amount,
            "| members=", len(s.member_fact_ids),
        )
