from __future__ import annotations

from bisect import bisect_left
from datetime import date
from decimal import Decimal


class FxConverter:
    """
    Deterministic fixed-rate currency conversion.

    Rates are supplied by the challenge dataset. No live FX is used.
    """

    def __init__(self, rates):
        self.index: dict[
            tuple[str, str],
            list[tuple[date, Decimal]]
        ] = {}

        for rate_date, frm, to, rate in rates:
            self.index.setdefault((frm, to), []).append(
                (rate_date, Decimal(rate))
            )

        for series in self.index.values():
            series.sort(key=lambda item: item[0])

    def _nearest(
        self,
        frm: str,
        to: str,
        on: date,
    ) -> Decimal | None:

        series = self.index.get((frm, to))

        if not series:
            return None

        dates = [d for d, _ in series]
        i = bisect_left(dates, on)

        candidates = []

        if i < len(series):
            candidates.append(series[i])

        if i > 0:
            candidates.append(series[i - 1])

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: abs((item[0] - on).days),
        )[1]

    def rate(
        self,
        frm: str,
        to: str,
        on: date,
    ) -> Decimal:

        if frm == to:
            return Decimal("1")

        # Direct rate.
        direct = self._nearest(frm, to, on)
        if direct is not None:
            return direct

        # Inverse rate.
        inverse = self._nearest(to, frm, on)
        if inverse is not None and inverse != 0:
            return Decimal("1") / inverse

        # USD bridge.
        if frm != "USD" and to != "USD":
            to_usd = self._nearest(frm, "USD", on)
            usd_to_target = self._nearest("USD", to, on)

            if (
                to_usd is not None
                and usd_to_target is not None
            ):
                return to_usd * usd_to_target

        raise ValueError(
            f"No FX path from {frm} to {to} around {on}"
        )

    def convert(
        self,
        amount: Decimal,
        frm: str,
        to: str,
        on: date,
    ) -> Decimal:

        if amount is None:
            return None

        return amount * self.rate(frm, to, on)