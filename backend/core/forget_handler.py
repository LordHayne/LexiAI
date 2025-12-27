import asyncio
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


FORGET_PATTERNS = [
    r"vergiss\s+(.+)",
    r"forget\s+(.+)",
    r"lösche\s+erinnerung(?:en)?\s+(?:an|über|zu)\s+(.+)",
    r"delete\s+memor(?:y|ies)\s+(?:about|of)\s+(.+)",
    r"vergiss\s+was\s+du\s+über\s+(.+)\s+weißt",
    r"forget\s+what\s+you\s+know\s+about\s+(.+)",
]


@dataclass
class ForgetResult:
    response: str
    deleted_ids: List[str]
    profile_updated: bool
    profile_cleared: bool

    @property
    def deleted_count(self) -> int:
        return len(self.deleted_ids)


def detect_forget_topic(message: str) -> Optional[str]:
    lowered = (message or "").lower()
    for pattern in FORGET_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            return match.group(1).strip()
    return None


def detect_profile_forget_keys(text: str) -> List[str]:
    lowered = (text or "").lower()
    if any(token in lowered for token in ["alles über mich", "alles ueber mich", "mein profil", "profil komplett"]):
        return ["*"]

    keys = []
    if any(token in lowered for token in ["name", "heiße", "heisse", "heiße ich"]):
        keys.append("user_profile_name")
    if any(token in lowered for token in ["beruf", "job", "arbeit", "profession"]):
        keys.append("user_profile_occupation")
    if any(token in lowered for token in ["hobby", "hobbys", "hobbies", "interesse", "interessen"]):
        keys.append("user_profile_interests")
    if any(token in lowered for token in ["skill", "skills", "fähigkeit", "fähigkeiten"]):
        keys.append("user_profile_skills")
    if any(token in lowered for token in ["sprache", "sprachen"]):
        keys.append("user_profile_languages")
    if any(token in lowered for token in ["präferenz", "praeferenz", "bevorzug", "mag es"]):
        keys.append("user_profile_preferences")
    if any(token in lowered for token in ["ziel", "ziele"]):
        keys.append("user_profile_goals")
    if any(token in lowered for token in ["wohnort", "stadt", "land"]):
        keys.append("user_profile_location")
    if any(token in lowered for token in ["technisch", "technik"]):
        keys.append("user_profile_technical_level")
    if any(token in lowered for token in ["kommunikation", "stil"]):
        keys.append("user_profile_communication_style")
    if any(token in lowered for token in ["hintergrund", "ausbildung"]):
        keys.append("user_profile_background")
    if any(token in lowered for token in ["themen", "topics"]):
        keys.append("user_profile_topics")
    return list(dict.fromkeys(keys))


def _profile_keys_to_fact_kinds(keys: List[str]) -> Set[str]:
    kinds: Set[str] = set()
    if "user_profile_name" in keys:
        kinds.add("name")
    if "user_profile_occupation" in keys:
        kinds.update({"occupation", "workplace"})
    if "user_profile_interests" in keys:
        kinds.add("interest")
    if "user_profile_skills" in keys:
        kinds.add("skill")
    if "user_profile_languages" in keys:
        kinds.add("language")
    if "user_profile_preferences" in keys:
        kinds.add("preference")
    if "user_profile_location" in keys:
        kinds.add("location")
    return kinds


def _build_forget_response(topic: str, deleted_count: int, is_german: bool,
                           profile_updated: bool, profile_cleared: bool) -> str:
    if deleted_count:
        if is_german:
            response = f"Ich habe {deleted_count} Erinnerung{'en' if deleted_count > 1 else ''} über '{topic}' gelöscht."
        else:
            response = f"I deleted {deleted_count} memor{'ies' if deleted_count > 1 else 'y'} about '{topic}'."
    else:
        if is_german:
            response = f"Ich habe keine Erinnerungen über '{topic}' gefunden."
        else:
            response = f"I found no memories about '{topic}'."

    if profile_updated:
        if is_german:
            response += " Dein Profil habe ich entsprechend bereinigt."
        else:
            response += " I also updated your profile accordingly."

    if profile_cleared:
        if is_german:
            response = response.replace("bereinigt", "zurückgesetzt")
        else:
            response = response.replace("updated", "reset")

    return response


async def apply_forget_by_topic(
    user_id: str,
    topic: str,
    is_german: bool,
    similarity_threshold: float = 0.70,
    profile_keys: Optional[List[str]] = None
) -> ForgetResult:
    from backend.memory.adapter import delete_memories_by_content, delete_user_facts
    from backend.services.user_store import get_user_store
    from backend.services.user_profile_qdrant import upsert_user_profile_in_qdrant

    deleted_ids = await asyncio.to_thread(
        delete_memories_by_content,
        query=topic,
        user_id=user_id,
        similarity_threshold=similarity_threshold
    )

    profile_updated = False
    profile_cleared = False

    if profile_keys:
        store = get_user_store()
        user = store.get_user(user_id)
        if user:
            profile = dict(user.profile) if user.profile else {}
            if profile_keys == ["*"]:
                profile = {}
                profile_cleared = True
            else:
                for key in profile_keys:
                    profile.pop(key, None)

            update_payload = {"profile": profile}
            if "user_profile_name" in profile_keys or profile_keys == ["*"]:
                update_payload["display_name"] = "Anonymous User"
            store.update_user(user_id, update_payload)
            await asyncio.to_thread(upsert_user_profile_in_qdrant, user_id, profile)
            profile_updated = True

        if profile_keys == ["*"]:
            await asyncio.to_thread(delete_user_facts, user_id)
        else:
            fact_kinds = _profile_keys_to_fact_kinds(profile_keys)
            for fact_kind in fact_kinds:
                await asyncio.to_thread(delete_user_facts, user_id, fact_kind)

    response = _build_forget_response(topic, len(deleted_ids), is_german, profile_updated, profile_cleared)
    return ForgetResult(
        response=response,
        deleted_ids=deleted_ids,
        profile_updated=profile_updated,
        profile_cleared=profile_cleared
    )
