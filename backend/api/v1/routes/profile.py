"""
User Profile API Routes

API Endpoints für User Profile Management und Automationsvorschläge.

Autor: LexiAI Development Team
Version: 1.0
Datum: 2025-01-23
"""

from fastapi import APIRouter, HTTPException
from typing import List
import logging

from backend.services.user_profile_manager import get_profile_manager
from backend.models.user_profile import (
    UserProfile, AutomationSuggestion, Routine
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str):
    """
    Hole User Profile.

    Args:
        user_id: User ID

    Returns:
        UserProfile
    """
    try:
        manager = get_profile_manager()
        profile = manager.load_profile(user_id)
        return profile

    except Exception as e:
        logger.error(f"Error loading profile for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{user_id}", response_model=UserProfile)
async def update_user_profile(user_id: str, profile: UserProfile):
    """
    Update User Profile.

    Args:
        user_id: User ID
        profile: UserProfile Daten

    Returns:
        Aktualisiertes UserProfile
    """
    try:
        # Sicherstellen dass user_id übereinstimmt
        if profile.user_id != user_id:
            raise HTTPException(
                status_code=400,
                detail="User ID in URL and profile must match"
            )

        manager = get_profile_manager()
        success = manager.save_profile(profile)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to save profile")

        return profile

    except Exception as e:
        logger.error(f"Error updating profile for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/suggestions", response_model=List[AutomationSuggestion])
async def get_automation_suggestions(
    user_id: str,
    min_confidence: float = 0.7
):
    """
    Hole Automationsvorschläge für User.

    Args:
        user_id: User ID
        min_confidence: Minimum Confidence (0-1)

    Returns:
        Liste von AutomationSuggestion
    """
    try:
        manager = get_profile_manager()
        suggestions = manager.generate_automation_suggestions(
            user_id=user_id,
            min_confidence=min_confidence
        )
        return suggestions

    except Exception as e:
        logger.error(f"Error generating suggestions for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{user_id}/suggestions/{suggestion_id}/accept")
async def accept_automation_suggestion(
    user_id: str,
    suggestion_id: str
):
    """
    Akzeptiere einen Automationsvorschlag und aktiviere die Routine.

    Args:
        user_id: User ID
        suggestion_id: Suggestion ID

    Returns:
        Erfolgs-Nachricht
    """
    try:
        manager = get_profile_manager()
        profile = manager.load_profile(user_id)

        # Finde Suggestion
        suggestion = None
        for s in profile.suggested_automations:
            if s.id == suggestion_id:
                suggestion = s
                break

        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        # Markiere als akzeptiert
        suggestion.accepted = True

        # Aktiviere die Routine
        if suggestion.routine.name == "Morgen-Routine":
            profile.morning_routine = suggestion.routine
            profile.morning_routine.enabled = True
        elif suggestion.routine.name == "Abend-Routine":
            profile.evening_routine = suggestion.routine
            profile.evening_routine.enabled = True
        elif suggestion.routine.name == "Nacht-Routine":
            profile.night_routine = suggestion.routine
            profile.night_routine.enabled = True
        else:
            # Custom Routine
            profile.custom_routines.append(suggestion.routine)

        manager.save_profile(profile)

        logger.info(f"✅ Suggestion {suggestion_id} accepted for {user_id}")

        return {
            "message": f"Automation '{suggestion.title}' wurde aktiviert",
            "routine_name": suggestion.routine.name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error accepting suggestion for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{user_id}/suggestions/{suggestion_id}/dismiss")
async def dismiss_automation_suggestion(
    user_id: str,
    suggestion_id: str
):
    """
    Lehne einen Automationsvorschlag ab.

    Args:
        user_id: User ID
        suggestion_id: Suggestion ID

    Returns:
        Erfolgs-Nachricht
    """
    try:
        manager = get_profile_manager()
        profile = manager.load_profile(user_id)

        # Finde Suggestion
        suggestion = None
        for s in profile.suggested_automations:
            if s.id == suggestion_id:
                suggestion = s
                break

        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")

        # Markiere als dismissed
        suggestion.dismissed = True
        manager.save_profile(profile)

        logger.info(f"✅ Suggestion {suggestion_id} dismissed for {user_id}")

        return {
            "message": f"Vorschlag '{suggestion.title}' wurde abgelehnt"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error dismissing suggestion for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/routines", response_model=List[Routine])
async def get_user_routines(user_id: str):
    """
    Hole alle Routinen für einen User.

    Args:
        user_id: User ID

    Returns:
        Liste von Routines
    """
    try:
        manager = get_profile_manager()
        profile = manager.load_profile(user_id)

        routines = []
        if profile.morning_routine:
            routines.append(profile.morning_routine)
        if profile.evening_routine:
            routines.append(profile.evening_routine)
        if profile.night_routine:
            routines.append(profile.night_routine)
        routines.extend(profile.custom_routines)

        return routines

    except Exception as e:
        logger.error(f"Error loading routines for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{user_id}/routines/{routine_name}/toggle")
async def toggle_routine(
    user_id: str,
    routine_name: str
):
    """
    Toggle (aktivieren/deaktivieren) eine Routine.

    Args:
        user_id: User ID
        routine_name: Name der Routine

    Returns:
        Neuer Status
    """
    try:
        manager = get_profile_manager()
        profile = manager.load_profile(user_id)

        # Finde Routine
        routine = None
        if profile.morning_routine and profile.morning_routine.name == routine_name:
            routine = profile.morning_routine
        elif profile.evening_routine and profile.evening_routine.name == routine_name:
            routine = profile.evening_routine
        elif profile.night_routine and profile.night_routine.name == routine_name:
            routine = profile.night_routine
        else:
            for r in profile.custom_routines:
                if r.name == routine_name:
                    routine = r
                    break

        if not routine:
            raise HTTPException(status_code=404, detail="Routine not found")

        # Toggle
        routine.enabled = not routine.enabled
        manager.save_profile(profile)

        logger.info(f"✅ Routine '{routine_name}' toggled for {user_id}: {routine.enabled}")

        return {
            "routine_name": routine_name,
            "enabled": routine.enabled,
            "message": f"Routine '{routine_name}' wurde {'aktiviert' if routine.enabled else 'deaktiviert'}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling routine for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/stats")
async def get_user_stats(user_id: str):
    """
    Hole Nutzungs-Statistiken für einen User.

    Args:
        user_id: User ID

    Returns:
        Statistik-Übersicht
    """
    try:
        manager = get_profile_manager()
        profile = manager.load_profile(user_id)

        # Sammle Stats
        total_rooms = len(profile.room_preferences)
        total_routines = sum([
            1 if profile.morning_routine else 0,
            1 if profile.evening_routine else 0,
            1 if profile.night_routine else 0,
            len(profile.custom_routines)
        ])
        active_routines = sum([
            1 if profile.morning_routine and profile.morning_routine.enabled else 0,
            1 if profile.evening_routine and profile.evening_routine.enabled else 0,
            1 if profile.night_routine and profile.night_routine.enabled else 0,
            sum(1 for r in profile.custom_routines if r.enabled)
        ])
        total_suggestions = len(profile.suggested_automations)
        active_suggestions = len(profile.get_active_suggestions())

        return {
            "user_id": user_id,
            "total_rooms_with_preferences": total_rooms,
            "total_routines": total_routines,
            "active_routines": active_routines,
            "total_suggestions": total_suggestions,
            "active_suggestions": active_suggestions,
            "automation_enabled": profile.automation_enabled,
            "learning_enabled": profile.learning_enabled,
            "last_updated": profile.last_updated.isoformat()
        }

    except Exception as e:
        logger.error(f"Error loading stats for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
