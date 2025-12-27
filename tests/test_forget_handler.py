import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from backend.core.forget_handler import (
    detect_forget_topic,
    detect_profile_forget_keys,
    apply_forget_by_topic,
)
from backend.models.user import User, UserTier


def _make_user(user_id="user-1"):
    now = datetime.now(timezone.utc).isoformat()
    return User(
        user_id=user_id,
        display_name="Thomas",
        created_at=now,
        last_seen=now,
        tier=UserTier.ANONYMOUS,
        preferences={},
        email=None,
        profile={"user_profile_name": "Thomas", "user_profile_interests": ["schach"]},
    )


def test_detect_forget_topic():
    assert detect_forget_topic("Vergiss meinen Namen") == "meinen namen"
    assert detect_forget_topic("delete memory about python") == "python"
    assert detect_forget_topic("kein forget hier") is None


def test_detect_profile_forget_keys():
    assert detect_profile_forget_keys("vergiss alles ueber mich") == ["*"]
    keys = detect_profile_forget_keys("vergiss meinen namen und beruf")
    assert "user_profile_name" in keys
    assert "user_profile_occupation" in keys


@pytest.mark.asyncio
async def test_apply_forget_clears_profile_and_facts(monkeypatch):
    deleted_ids = ["id-1", "id-2"]
    delete_memories = MagicMock(return_value=deleted_ids)
    delete_user_facts = MagicMock(return_value=2)
    upsert_profile = MagicMock(return_value="profile:user-1")

    user = _make_user()

    class DummyStore:
        def __init__(self):
            self.updated = None

        def get_user(self, _user_id):
            return user

        def update_user(self, _user_id, updates):
            self.updated = updates
            return user

    store = DummyStore()

    monkeypatch.setattr("backend.memory.adapter.delete_memories_by_content", delete_memories)
    monkeypatch.setattr("backend.memory.adapter.delete_user_facts", delete_user_facts)
    monkeypatch.setattr("backend.services.user_store.get_user_store", lambda: store)
    monkeypatch.setattr("backend.services.user_profile_qdrant.upsert_user_profile_in_qdrant", upsert_profile)

    result = await apply_forget_by_topic(
        user_id="user-1",
        topic="alles ueber mich",
        is_german=True,
        similarity_threshold=0.70,
        profile_keys=["*"],
    )

    assert result.deleted_ids == deleted_ids
    assert store.updated["profile"] == {}
    assert store.updated["display_name"] == "Anonymous User"
    delete_user_facts.assert_called_once_with("user-1")
    upsert_profile.assert_called_once()


@pytest.mark.asyncio
async def test_apply_forget_specific_keys(monkeypatch):
    delete_memories = MagicMock(return_value=[])
    delete_user_facts = MagicMock(return_value=1)
    upsert_profile = MagicMock(return_value="profile:user-1")
    user = _make_user()

    class DummyStore:
        def __init__(self):
            self.updated = None

        def get_user(self, _user_id):
            return user

        def update_user(self, _user_id, updates):
            self.updated = updates
            return user

    store = DummyStore()

    monkeypatch.setattr("backend.memory.adapter.delete_memories_by_content", delete_memories)
    monkeypatch.setattr("backend.memory.adapter.delete_user_facts", delete_user_facts)
    monkeypatch.setattr("backend.services.user_store.get_user_store", lambda: store)
    monkeypatch.setattr("backend.services.user_profile_qdrant.upsert_user_profile_in_qdrant", upsert_profile)

    result = await apply_forget_by_topic(
        user_id="user-1",
        topic="mein name",
        is_german=True,
        similarity_threshold=0.70,
        profile_keys=["user_profile_name", "user_profile_interests"],
    )

    assert result.deleted_ids == []
    assert store.updated["display_name"] == "Anonymous User"
    assert "user_profile_name" not in store.updated["profile"]
    assert "user_profile_interests" not in store.updated["profile"]
    calls = {call.args[1] for call in delete_user_facts.call_args_list}
    assert calls == {"name", "interest"}
