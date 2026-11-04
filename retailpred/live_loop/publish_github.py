"""Publish the public snapshot to the machine-written ``live-data`` branch.

GitHub Pages is the presentation layer; this module only updates one JSON
file on a dedicated branch through the Contents API. The credential never
enters log output, and identical snapshots are skipped so the branch history
records only real pipeline runs.
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
    """Create, update, or skip the snapshot file on the data branch."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
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
