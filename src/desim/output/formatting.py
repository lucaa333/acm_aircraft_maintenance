from __future__ import annotations

from desim.domain.metrics import ModelResult


def format_results(results: list[ModelResult]) -> str:
    # Builds a simple fixed-width text table for terminal output.
    rows = [result.as_dict() for result in results]
    headers = list(rows[0].keys()) if rows else []
    widths = {header: max(len(header), *(len(str(row.get(header, ""))) for row in rows)) for header in headers}
    lines = []
    lines.append(" | ".join(header.ljust(widths[header]) for header in headers))
    lines.append("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        lines.append(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))
    return "\n".join(lines)
