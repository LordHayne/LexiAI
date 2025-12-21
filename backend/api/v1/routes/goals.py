"""
Goals API Endpoints

Verwaltet Benutzerziele über REST API.
"""

import logging
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.memory.goal_tracker import (
    get_goal_tracker,
    Goal,
    GoalStatus,
    GoalPriority
)
from backend.core.component_cache import get_cached_components

logger = logging.getLogger("lexi_middleware.goals_api")
router = APIRouter()


# Request/Response Models
class GoalCreateRequest(BaseModel):
    """Request model für neues Ziel"""
    content: str = Field(..., description="Beschreibung des Ziels")
    category: str = Field(default="general", description="Kategorie (health, learning, work, etc.)")
    priority: str = Field(default="medium", description="Priorität (low, medium, high, urgent)")
    target_date: Optional[str] = Field(None, description="Zieldatum (ISO format)")
    metadata: Optional[dict] = Field(default={}, description="Zusätzliche Metadaten")


class GoalUpdateRequest(BaseModel):
    """Request model für Ziel-Update"""
    content: Optional[str] = Field(None, description="Neue Beschreibung")
    category: Optional[str] = Field(None, description="Neue Kategorie")
    priority: Optional[str] = Field(None, description="Neue Priorität")
    status: Optional[str] = Field(None, description="Neuer Status")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Fortschritt (0.0-1.0)")
    target_date: Optional[str] = Field(None, description="Neues Zieldatum")


class GoalResponse(BaseModel):
    """Response model für Ziel"""
    id: str
    user_id: str
    content: str
    category: str
    status: str
    priority: str
    progress: float
    created_at: str
    updated_at: str
    target_date: Optional[str] = None
    mentions: int
    last_mentioned: Optional[str] = None
    milestones: List[str] = []
    source_memory_ids: List[str] = []
    metadata: dict = {}


def goal_to_response(goal: Goal) -> GoalResponse:
    """Konvertiert Goal zu Response Model"""
    return GoalResponse(
        id=goal.id,
        user_id=goal.user_id,
        content=goal.content,
        category=goal.category,
        status=goal.status.value,
        priority=goal.priority.value,
        progress=goal.progress,
        created_at=goal.created_at.isoformat(),
        updated_at=goal.updated_at.isoformat(),
        target_date=goal.target_date.isoformat() if goal.target_date else None,
        mentions=goal.mentions,
        last_mentioned=goal.last_mentioned.isoformat() if goal.last_mentioned else None,
        milestones=goal.milestones,
        source_memory_ids=goal.source_memory_ids,
        metadata=goal.metadata
    )


@router.get("/goals", response_model=List[GoalResponse])
async def get_goals(
    user_id: str = Query(default="default", description="User ID"),
    status: Optional[str] = Query(None, description="Filter by status (active, completed, abandoned, paused)")
):
    """
    Holt alle Ziele eines Benutzers.

    Optional: Filter nach Status.
    """
    try:
        bundle = get_cached_components()
        tracker = get_goal_tracker(bundle.vectorstore)

        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = GoalStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of: active, completed, abandoned, paused"
                )

        goals = tracker.get_all_goals(user_id, status=status_filter)

        logger.info(f"Retrieved {len(goals)} goals for user {user_id}")

        return [goal_to_response(g) for g in goals]

    except Exception as e:
        logger.error(f"Error retrieving goals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/goals/{goal_id}", response_model=GoalResponse)
async def get_goal(
    goal_id: str,
    user_id: str = Query(default="default", description="User ID")
):
    """Holt ein einzelnes Ziel"""
    try:
        bundle = get_cached_components()
        tracker = get_goal_tracker(bundle.vectorstore)

        # Get all goals and find the one with matching ID
        goals = tracker.get_all_goals(user_id)
        goal = next((g for g in goals if g.id == goal_id), None)

        if not goal:
            raise HTTPException(status_code=404, detail=f"Goal {goal_id} not found")

        return goal_to_response(goal)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving goal {goal_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/goals", response_model=GoalResponse, status_code=201)
async def create_goal(
    request: GoalCreateRequest,
    user_id: str = Query(default="default", description="User ID")
):
    """Erstellt ein neues Ziel"""
    try:
        bundle = get_cached_components()
        tracker = get_goal_tracker(bundle.vectorstore)

        # Parse priority
        try:
            priority = GoalPriority(request.priority.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid priority. Must be one of: low, medium, high, urgent"
            )

        # Parse target date
        target_date = None
        if request.target_date:
            try:
                target_date = datetime.fromisoformat(request.target_date)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid target_date format. Use ISO format (YYYY-MM-DD)"
                )

        # Create goal
        from uuid import uuid4
        goal = Goal(
            id=str(uuid4()),
            user_id=user_id,
            content=request.content,
            category=request.category,
            status=GoalStatus.ACTIVE,
            priority=priority,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            target_date=target_date,
            mentions=0,
            metadata=request.metadata or {}
        )

        success = tracker.add_goal(goal)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to create goal")

        logger.info(f"Created goal {goal.id} for user {user_id}: {request.content[:50]}...")

        return goal_to_response(goal)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating goal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/goals/{goal_id}", response_model=GoalResponse)
async def update_goal(
    goal_id: str,
    request: GoalUpdateRequest,
    user_id: str = Query(default="default", description="User ID")
):
    """Aktualisiert ein Ziel"""
    try:
        bundle = get_cached_components()
        tracker = get_goal_tracker(bundle.vectorstore)

        # Get existing goal
        goals = tracker.get_all_goals(user_id)
        goal = next((g for g in goals if g.id == goal_id), None)

        if not goal:
            raise HTTPException(status_code=404, detail=f"Goal {goal_id} not found")

        # Update fields
        if request.content is not None:
            goal.content = request.content

        if request.category is not None:
            goal.category = request.category

        if request.priority is not None:
            try:
                goal.priority = GoalPriority(request.priority.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid priority. Must be one of: low, medium, high, urgent"
                )

        if request.status is not None:
            try:
                goal.status = GoalStatus(request.status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of: active, completed, abandoned, paused"
                )

        if request.progress is not None:
            goal.progress = request.progress

        if request.target_date is not None:
            try:
                goal.target_date = datetime.fromisoformat(request.target_date)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid target_date format. Use ISO format (YYYY-MM-DD)"
                )

        # Update timestamp
        goal.updated_at = datetime.now(timezone.utc)

        success = tracker.update_goal(goal)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update goal")

        logger.info(f"Updated goal {goal_id}")

        return goal_to_response(goal)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating goal {goal_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/goals/{goal_id}", status_code=204)
async def delete_goal(
    goal_id: str,
    user_id: str = Query(default="default", description="User ID")
):
    """Löscht ein Ziel"""
    try:
        bundle = get_cached_components()
        tracker = get_goal_tracker(bundle.vectorstore)

        # Verify goal exists and belongs to user
        goals = tracker.get_all_goals(user_id)
        goal = next((g for g in goals if g.id == goal_id), None)

        if not goal:
            raise HTTPException(status_code=404, detail=f"Goal {goal_id} not found")

        success = tracker.delete_goal(goal_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete goal")

        logger.info(f"Deleted goal {goal_id}")

        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting goal {goal_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/goals/{goal_id}/complete", response_model=GoalResponse)
async def complete_goal(
    goal_id: str,
    user_id: str = Query(default="default", description="User ID")
):
    """Markiert ein Ziel als abgeschlossen"""
    try:
        bundle = get_cached_components()
        tracker = get_goal_tracker(bundle.vectorstore)

        # Get existing goal
        goals = tracker.get_all_goals(user_id)
        goal = next((g for g in goals if g.id == goal_id), None)

        if not goal:
            raise HTTPException(status_code=404, detail=f"Goal {goal_id} not found")

        # Update status and progress
        goal.status = GoalStatus.COMPLETED
        goal.progress = 1.0
        goal.updated_at = datetime.now(timezone.utc)

        success = tracker.update_goal(goal)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to complete goal")

        logger.info(f"Completed goal {goal_id}")

        return goal_to_response(goal)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing goal {goal_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/goals/stats/summary")
async def get_goals_summary(
    user_id: str = Query(default="default", description="User ID")
):
    """
    Gibt Zusammenfassung der Ziele zurück.

    Statistiken über aktive, abgeschlossene, etc. Ziele.
    """
    try:
        bundle = get_cached_components()
        tracker = get_goal_tracker(bundle.vectorstore)

        all_goals = tracker.get_all_goals(user_id)

        # Count by status
        stats = {
            "total": len(all_goals),
            "active": sum(1 for g in all_goals if g.status == GoalStatus.ACTIVE),
            "completed": sum(1 for g in all_goals if g.status == GoalStatus.COMPLETED),
            "abandoned": sum(1 for g in all_goals if g.status == GoalStatus.ABANDONED),
            "paused": sum(1 for g in all_goals if g.status == GoalStatus.PAUSED),
        }

        # Count by category
        categories = {}
        for goal in all_goals:
            categories[goal.category] = categories.get(goal.category, 0) + 1

        # Count by priority
        priorities = {}
        for goal in all_goals:
            priorities[goal.priority.value] = priorities.get(goal.priority.value, 0) + 1

        # Average progress (nur aktive Goals)
        active_goals = [g for g in all_goals if g.status == GoalStatus.ACTIVE]
        avg_progress = sum(g.progress for g in active_goals) / len(active_goals) if active_goals else 0.0

        return {
            "success": True,
            "user_id": user_id,
            "stats": stats,
            "categories": categories,
            "priorities": priorities,
            "average_progress": round(avg_progress, 2)
        }

    except Exception as e:
        logger.error(f"Error getting goals summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
