from datetime import date
from decimal import Decimal

from loader import load_dataset


d = load_dataset("../dataset")

checks = [
    ("user_02", date(2025, 8, 5)),
    ("user_05", date(2025, 11, 5)),
    ("user_10", date(2024, 11, 6)),
    ("user_13", date(2024, 3, 6)),
    ("user_20", date(2024, 3, 6)),
    ("user_25", date(2024, 3, 6)),
]


for uid, req_date in checks:
    profile = d.profiles[uid]
    events = d.events_by_user[uid]

    balance = Decimal("0")
    credits = Decimal("0")
    debits = Decimal("0")
    count = 0

    for e in events:
        if e.event_date > req_date:
            continue

        if e.settlement_date is not None and e.settlement_date > req_date:
            continue

        if e.status.value != "settled":
            continue

        if e.direction.value == "credit":
            balance += e.amount
            credits += e.amount
            count += 1

        elif e.direction.value == "debit":
            balance -= e.amount
            debits += e.amount
            count += 1

    profile_balance = profile.current_available_balance

    print("=" * 70)
    print(f"{uid}  request_date={req_date}")
    print(f"profile current balance : {profile_balance}")
    print(f"settled credits        : {credits}")
    print(f"settled debits         : {debits}")
    print(f"ledger net             : {balance}")
    print(f"difference             : {profile_balance - balance}")
    print(f"events included        : {count}")