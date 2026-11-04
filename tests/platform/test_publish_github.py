import base64
import hashlib

import pytest

from retailpred.live_loop.publish_github import commit_snapshot, content_sha256

TOKEN = "ghp_test-token-value"
CONTENT = b'{"contract_version": "2.1"}'
MESSAGE = "live snapshot from pipeline run run-123"


class FakeResponse:
    def __init__(self, status_code: int, payload=None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload or {}


class FakeSession:
    def __init__(self, get_response: FakeResponse, put_response: FakeResponse | None = None) -> None:
        self.get_response = get_response
        self.put_response = put_response or FakeResponse(201, {"commit": {"sha": "abc123def456"}})
        self.gets: list[dict] = []
        self.puts: list[dict] = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.gets.append({"url": url, "headers": headers, "params": params})
        return self.get_response

    def put(self, url, headers=None, json=None, timeout=None):
        self.puts.append({"url": url, "headers": headers, "json": json})
        return self.put_response


def _existing_response(content: bytes) -> FakeResponse:
    return FakeResponse(
        200,
        {
            "sha": "existing-blob-sha",
            "content": base64.b64encode(content).decode("ascii"),
        },
    )


def test_creates_file_when_branch_has_no_snapshot() -> None:
    session = FakeSession(FakeResponse(404))
    result = commit_snapshot(
        session,
        repo="oleeveeuh/retailPRED",
        branch="live-data",
        path="live/v2-latest.json",
        token=TOKEN,
        content=CONTENT,
        message=MESSAGE,
    )
    assert result.action == "created"
    assert len(session.puts) == 1
    body = session.puts[0]["json"]
    assert body["message"] == MESSAGE
    assert body["branch"] == "live-data"
    assert "sha" not in body
    assert base64.b64decode(body["content"]) == CONTENT


def test_updates_when_content_differs() -> None:
    session = FakeSession(_existing_response(b'{"old": true}'))
    result = commit_snapshot(
        session,
        repo="oleeveeuh/retailPRED",
        branch="live-data",
        path="live/v2-latest.json",
        token=TOKEN,
        content=CONTENT,
        message=MESSAGE,
    )
    assert result.action == "updated"
    body = session.puts[0]["json"]
    assert body["sha"] == "existing-blob-sha"


def test_skips_identical_content_without_committing() -> None:
    session = FakeSession(_existing_response(CONTENT))
    result = commit_snapshot(
        session,
        repo="oleeveeuh/retailPRED",
        branch="live-data",
        path="live/v2-latest.json",
        token=TOKEN,
        content=CONTENT,
        message=MESSAGE,
    )
    assert result.action == "skipped"
    assert result.commit_sha is None
    assert session.puts == []


def test_unexpected_lookup_failure_raises_without_committing() -> None:
    session = FakeSession(FakeResponse(500))
    with pytest.raises(RuntimeError, match="lookup failed"):
        commit_snapshot(
            session,
            repo="oleeveeuh/retailPRED",
            branch="live-data",
            path="live/v2-latest.json",
            token=TOKEN,
            content=CONTENT,
            message=MESSAGE,
        )
    assert session.puts == []


def test_token_stays_out_of_urls_and_params() -> None:
    session = FakeSession(FakeResponse(404))
    commit_snapshot(
        session,
        repo="oleeveeuh/retailPRED",
        branch="live-data",
        path="live/v2-latest.json",
        token=TOKEN,
        content=CONTENT,
        message=MESSAGE,
    )
    for call in session.gets + session.puts:
        assert TOKEN not in call["url"]
        serialized = str(call.get("params")) + str(call.get("json"))
        assert TOKEN not in serialized
        assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_content_hash_is_sha256() -> None:
    assert content_sha256(CONTENT) == hashlib.sha256(CONTENT).hexdigest()
