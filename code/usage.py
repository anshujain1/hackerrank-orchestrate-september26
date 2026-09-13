
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class UsageTracker:
    """Track model/API usage for the hackathon usage report."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    errors: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        error: bool = False,
        **kwargs: Any,
    ) -> None:
        self.calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.cached_input_tokens += int(cached_input_tokens or 0)

        if error:
            self.errors += 1

        self.details.append(
            {
                "model": model or "",
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "cached_input_tokens": int(cached_input_tokens or 0),
                "error": bool(error),
                **kwargs,
            }
        )

    def add_usage(self, usage: Any, **kwargs: Any) -> None:
        """Accept common API usage objects or dictionaries."""

        if usage is None:
            return

        if isinstance(usage, dict):
            input_tokens = usage.get(
                "input_tokens",
                usage.get("prompt_token_count", 0),
            )
            output_tokens = usage.get(
                "output_tokens",
                usage.get("candidates_token_count", 0),
            )
            cached_input_tokens = usage.get(
                "cache_read_input_tokens",
                usage.get(
                    "cached_input_tokens",
                    usage.get("cached_content_token_count", 0),
                ),
            )
        else:
            input_tokens = getattr(
                usage,
                "input_tokens",
                getattr(usage, "prompt_token_count", 0),
            )
            output_tokens = getattr(
                usage,
                "output_tokens",
                getattr(usage, "candidates_token_count", 0),
            )
            cached_input_tokens = getattr(
                usage,
                "cache_read_input_tokens",
                getattr(
                    usage,
                    "cached_input_tokens",
                    getattr(usage, "cached_content_token_count", 0),
                ),
            )

        self.record(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            **kwargs,
        )

    def save(self, path: str, num_requests: int | None = None) -> None:
        """Write a human-readable usage report."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        request_count = (
            num_requests
            if num_requests is not None
            else 0
        )

        lines = [
            "# Usage Report",
            "",
            "## Summary",
            "",
            f"- Requests processed: {request_count}",
            f"- Model/API calls: {self.calls}",
            f"- Input tokens: {self.input_tokens}",
            f"- Output tokens: {self.output_tokens}",
            f"- Cached input tokens: {self.cached_input_tokens}",
            f"- Errors: {self.errors}",
            "",
        ]

        if self.details:
            lines.extend(
                [
                    "## Call Details",
                    "",
                    "| # | Model | Input tokens | Output tokens | Cached input tokens | Error |",
                    "|---:|---|---:|---:|---:|---|",
                ]
            )

            for index, detail in enumerate(self.details, start=1):
                lines.append(
                    "| "
                    f"{index} | "
                    f"{detail.get('model', '')} | "
                    f"{detail.get('input_tokens', 0)} | "
                    f"{detail.get('output_tokens', 0)} | "
                    f"{detail.get('cached_input_tokens', 0)} | "
                    f"{detail.get('error', False)} |"
                )

            lines.append("")

        output_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

