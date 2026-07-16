from datetime import datetime
from bson import ObjectId
from app.data.mongo_client import get_db

COLLECTION_LOGS = "training_logs"
COLLECTION_PLANS = "training_plans"


def _logs():
    return get_db()[COLLECTION_LOGS]


def _plans():
    return get_db()[COLLECTION_PLANS]


def create_plan(user_id: str, plan_data: dict) -> dict:
    plan_data["user_id"] = user_id
    plan_data["status"] = "active"
    plan_data["created_at"] = datetime.utcnow().isoformat()
    result = _plans().insert_one(plan_data)
    plan_data["_id"] = str(result.inserted_id)
    return plan_data


def get_active_plan(user_id: str) -> dict | None:
    return _plans().find_one({"user_id": user_id, "status": "active"})


def deactivate_plan(plan_id: str):
    _plans().update_one(
        {"_id": ObjectId(plan_id)},
        {"$set": {"status": "inactive"}},
    )


def create_training_log(user_id: str, log_data: dict) -> str:
    log_data["user_id"] = user_id
    log_data["date"] = datetime.utcnow().isoformat()
    result = _logs().insert_one(log_data)
    return str(result.inserted_id)


def get_training_logs(
    user_id: str, limit: int = 30, skip: int = 0
) -> list[dict]:
    return list(
        _logs()
        .find({"user_id": user_id})
        .sort("date", -1)
        .skip(skip)
        .limit(limit)
    )


def get_log_by_id(log_id: str) -> dict | None:
    return _logs().find_one({"_id": ObjectId(log_id)})
