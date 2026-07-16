import json
from pathlib import Path
from app.data.mongo_client import get_db
from loguru import logger

SEED_FILE = Path(__file__).parent.parent.parent / "seeds" / "exercise_seed.json"
COLLECTION = "exercises"


def _col():
    return get_db()[COLLECTION]


def seed_exercises(force: bool = False):
    if _col().count_documents({}) > 0 and not force:
        logger.info(f"{_col().count_documents({})} exercises already seeded")
        return
    if force:
        _col().delete_many({})
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        exercises = json.load(f)
    _col().insert_many(exercises)
    logger.info(f"Seeded {len(exercises)} exercises")


def get_exercise_by_id(exercise_id: str) -> dict | None:
    return _col().find_one({"exercise_id": exercise_id})


def search_exercises(
    primary_muscle: str | None = None,
    equipment: str | None = None,
    difficulty: int | None = None,
    category: str | None = None,
    limit: int = 20,
) -> list[dict]:
    query = {}
    if primary_muscle:
        query["primary_muscles"] = primary_muscle
    if equipment:
        query["equipment"] = equipment
    if difficulty is not None:
        query["difficulty"] = difficulty
    if category:
        query["category"] = category
    return list(_col().find(query).limit(limit))


def list_all_exercises() -> list[dict]:
    return list(_col().find({}))
