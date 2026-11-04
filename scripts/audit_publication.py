"""Fail fast when the recruiter-facing publication surface regresses."""

from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FRONTEND = (
    ROOT / "frontend/src/App.tsx",
    ROOT / "frontend/src/content/showcase.ts",
    ROOT / "frontend/src/components/DemoBanner.tsx",
    ROOT / "frontend/src/components/LatestForecastPanel.tsx",
    ROOT / "frontend/src/components/layout/Header.tsx",
    ROOT / "frontend/src/components/layout/Sidebar.tsx",
    ROOT / "frontend/src/pages/PredictionsPage.tsx",
    ROOT / "frontend/src/pages/ShowcasePage.tsx",
    ROOT / "frontend/src/api/v2Forecasts.ts",
)


def markdown_link_errors(path: Path) -> list[str]:
    errors: list[str] = []
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("#"):
            continue
        local_target = target.split("#", 1)[0]
        if local_target and not (path.parent / local_target).exists():
            errors.append(f"{path.relative_to(ROOT)} links to missing {target}")
    return errors


def _dockerignore_includes(lines: list[str], relpath: str) -> bool:
    """Approximate Docker's last-match-wins context filter for the rule shapes used here."""
    included = True
    for raw in lines:
        rule = raw.strip()
        if not rule or rule.startswith("#"):
            continue
        negated = rule.startswith("!")
        pattern = rule.lstrip("!").rstrip("/")
        if pattern == "*":
            matched = "/" not in relpath
        elif pattern.endswith("/**"):
            matched = relpath.startswith(pattern[:-3] + "/")
        elif "*" in pattern:
            matched = fnmatch.fnmatch(relpath, pattern)
        else:
            matched = relpath == pattern or relpath.startswith(pattern + "/")
        if matched:
            # Last matching rule wins: negations re-include, plain patterns exclude.
            included = negated
    return included


def dockerignore_errors() -> list[str]:
    """The serving image must contain the v2 contract and exclude legacy artifacts."""
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    probes: dict[str, bool] = {
        "backend/main.py": True,
        "backend/api/v2_forecasts.py": True,
        "backend/demo/v2_forecasts.json": True,
        "forecasting/portfolio.py": True,
        "requirements-serving.txt": True,
        "backend/ml/models/automobile_dealers_LGBM_model.pkl": False,
    }
    errors = []
    for relpath, expected in probes.items():
        included = _dockerignore_includes(lines, relpath)
        if included is not expected:
            errors.append(
                f".dockerignore {'includes' if included else 'excludes'} {relpath} "
                f"(must be {'in the serving image' if expected else 'excluded from it'})"
            )
    return errors


def main() -> int:
    errors = markdown_link_errors(ROOT / "README.md")
    errors += dockerignore_errors()

    active_ui = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_FRONTEND)
    forbidden_ui_claims = {
        "production database": "unsupported production-database claim",
        "currently deployed": "unsupported deployment claim",
        "0.26%": "unsupported legacy metric",
        "90.2": "unsupported legacy accuracy",
        "/dashboard/overview": "legacy overview route is public",
        "/dashboard/models": "legacy models route is public",
    }
    for phrase, message in forbidden_ui_claims.items():
        if phrase.lower() in active_ui.lower():
            errors.append(message)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in ("v2 rebuild active: Milestone 1", "current API and dashboard still serve v1"):
        if phrase.lower() in readme.lower():
            errors.append(f"README contains stale claim: {phrase}")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for required in ("backend.main:app", "/api/v2/health", "USER appuser"):
        if required not in dockerfile:
            errors.append(f"Dockerfile missing publication requirement: {required}")

    workbook = (ROOT / "bi/tableau/retailpred_v2.twb").read_text(encoding="utf-8")
    if 'dbname="retailpred_v2.hyper"' not in workbook:
        errors.append("Tableau workbook does not use the portable relative Hyper filename")

    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github/workflows").glob("*.yml"))
    )
    for phrase in ("node-version: '20'", "gitleaks/gitleaks-action@v2", "@master"):
        if phrase in workflows:
            errors.append(f"workflow contains obsolete or mutable action reference: {phrase}")

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    real_envs = [path for path in tracked if path.endswith("/.env") or path == ".env"]
    if real_envs:
        errors.append(f"tracked non-template environment files: {', '.join(real_envs)}")

    if errors:
        print("Publication audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    pickle_count = sum(path.endswith(".pkl") for path in tracked)
    print("Publication audit passed.")
    print("Serving image context includes the v2 contract and excludes legacy model artifacts.")
    print(f"Known debt: {pickle_count} tracked legacy pickle files remain isolated from v2 serving.")
    if "predictions.csv" in tracked:
        print("Known debt: tracked legacy predictions.csv is not a v2 evidence source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
