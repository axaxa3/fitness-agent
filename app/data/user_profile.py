from datetime import datetime
from app.data.mongo_client import get_db

COLLECTION = "user_profiles"


def _col():
    return get_db()[COLLECTION]


def create_profile(user_id: str, profile_data: dict) -> dict:
    doc = {
        "user_id": user_id,
        "basic": profile_data.get("basic", {}),
        "fitness_profile": profile_data.get("fitness_profile", {}),
        "strength_snapshot": {},
        "muscle_balance": {},
        "injury_flags": [],
        "training_age_weeks": 0,
        "fatigue_trend": "normal",
        "recent_rpe_avg": 0.0,
        "onboarding_completed": False,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    _col().insert_one(doc)
    return doc


def get_profile(user_id: str) -> dict | None:
    return _col().find_one({"user_id": user_id})


def update_profile(user_id: str, updates: dict) -> bool:
    updates["updated_at"] = datetime.utcnow().isoformat()
    result = _col().update_one(
        {"user_id": user_id},
        {"$set": updates},
    )
    return result.modified_count > 0


def mark_onboarding_complete(user_id: str) -> bool:
    return update_profile(user_id, {"onboarding_completed": True})
