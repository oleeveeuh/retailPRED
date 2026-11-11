"""Publish machine-written artifacts to the ``live-data`` branch.

GitHub Pages is the presentation layer; this module only updates JSON files
on a dedicated branch through the Contents API — the forecast snapshot and,
since contract ``2.2-status``, the per-run ingest-status record and the
append-only ``runs.jsonl`` log. The credential never enters log output, and
identical content is skipped so the branch history records only real
pipeline runs.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any

import requests

CONTENTS_URL = "https://api.github.com/repos/{repo}/contents/{path}"


@dataclass(frozen=True)
class PublishResult:
    action: str  # "created", "updated", or "skipped"
    commit_sha: str | None


def content_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def fetch_content(
    session: requests.Session,
    *,
    repo: str,
    branch: str,
    path: str,
    token: str,
) -> bytes | None:
    """Read a file from the data branch; None when it does not exist yet."""
    url = CONTENTS_URL.format(repo=repo, path=path)
    response = session.get(url, headers=_headers(token), params={"ref": branch}, timeout=30)
    if response.status_code == 200:
        return base64.b64decode(response.json().get("content"))
    if response.status_code == 404:
        return None
    raise RuntimeError(
        f"contents lookup for {path} failed with HTTP {response.status_code}; "
        "check the publisher credential scope"
    )


def append_run_line(
    session: requests.Session,
    *,
    repo: str,
    branch: str,
    path: str,
    token: str,
    line: str,
    message: str,
) -> PublishResult:
    """Append one JSON line to the per-run log by rewriting prior lines + the new line.

    The Contents API has no append, so the file is fetched, extended, and
    committed whole; an exact replay of the last recorded line leaves the
    content unchanged so commit_snapshot's identical-hash skip makes it a
    no-op. The file is never truncated.
    """
    existing = fetch_content(session, repo=repo, branch=branch, path=path, token=token)
    prior = existing.decode("utf-8").rstrip("\n") if existing else ""
    lines = prior.split("\n") if prior else []
    if lines and lines[-1] == line:
        content = existing
    else:
        content = ("\n".join(lines + [line]) + "\n").encode("utf-8")
    return commit_snapshot(
        session,
        repo=repo,
        branch=branch,
        path=path,
        token=token,
        content=content,
        message=message,
    )


def commit_snapshot(
    session: requests.Session,
    *,
    repo: str,
    branch: str,
    path: str,
    token: str,
    content: bytes,
    message: str,
) -> PublishResult:
    """Create, update, or skip a file on the data branch."""
    headers = _headers(token)
    url = CONTENTS_URL.format(repo=repo, path=path)
    existing = session.get(url, headers=headers, params={"ref": branch}, timeout=30)
    existing_sha: str | None = None
    if existing.status_code == 200:
        existing_payload = existing.json()
        existing_sha = existing_payload.get("sha")
        existing_content = existing_payload.get("content")
        if existing_content is not None:
            decoded = base64.b64decode(existing_content)
            if content_sha256(decoded) == content_sha256(content):
                return PublishResult(action="skipped", commit_sha=None)
    elif existing.status_code != 404:
        raise RuntimeError(
            f"snapshot lookup failed with HTTP {existing.status_code}; "
            "check the publisher credential scope"
        )

    body: dict[str, Any] = {
        "message": message,
        "branch": branch,
        "content": base64.b64encode(content).decode("ascii"),
    }
    if existing_sha is not None:
        body["sha"] = existing_sha
    response = session.put(url, headers=headers, json=body, timeout=30)
    if response.status_code not in {200, 201}:
        raise RuntimeError(f"snapshot commit failed with HTTP {response.status_code}")
    action = "updated" if existing_sha is not None else "created"
    commit_sha = response.json().get("commit", {}).get("sha")
    return PublishResult(action=action, commit_sha=commit_sha)
