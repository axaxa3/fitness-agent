from pymilvus import Collection, CollectionSchema, DataType, FieldSchema
from loguru import logger

from app.conf.settings import config
from app.core.llm_factory import create_embeddings
from app.core.milvus_client import milvus_manager
from app.data.mongo_client import get_db

COLLECTION_NAME = "exercise_kb"
DIM = config.embedding_dim

EXERCISE_SCHEMA = CollectionSchema([
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="exercise_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="name_cn", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="primary_muscles", dtype=DataType.JSON),
    FieldSchema(name="equipment", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="difficulty", dtype=DataType.INT64),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
])

INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}


def create_exercise_collection():
    milvus_manager.create_collection_if_not_exists(
        COLLECTION_NAME, EXERCISE_SCHEMA, INDEX_PARAMS,
    )


def index_all_exercises(force: bool = False):
    col = milvus_manager.get_collection(COLLECTION_NAME)
    if col.num_entities > 0 and not force:
        logger.info(f"Exercise KB already has {col.num_entities} entities")
        return

    if force:
        col.delete("id >= 0")

    embeddings = create_embeddings()
    mongo_exercises = list(get_db()["exercises"].find({}))

    texts = []
    entities = []
    for ex in mongo_exercises:
        search_text = (
            f"{ex['name_cn']} {ex['name']}。"
            f"目标肌群：{'、'.join(ex['primary_muscles'])}。"
            f"{ex.get('tips', '')}"
        )
        texts.append(search_text)
        entities.append({
            "exercise_id": ex["exercise_id"],
            "name": ex["name"],
            "name_cn": ex["name_cn"],
            "text": search_text,
            "primary_muscles": ex["primary_muscles"],
            "equipment": ex["equipment"],
            "difficulty": ex["difficulty"],
        })

    vectors = embeddings.embed_documents(texts)

    insert_data = []
    for i, ent in enumerate(entities):
        insert_data.append({**ent, "vector": vectors[i]})

    col.insert(insert_data)
    col.flush()
    logger.info(f"Indexed {len(entities)} exercises to Milvus")


def search_exercises_semantic(query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
    col = milvus_manager.get_collection(COLLECTION_NAME)
    embeddings = create_embeddings()
    query_vector = embeddings.embed_query(query)

    expr = None
    if filters:
        conditions = []
        if "equipment" in filters:
            conditions.append(f'equipment == "{filters["equipment"]}"')
        if "difficulty" in filters:
            conditions.append(f"difficulty == {filters['difficulty']}")
        if conditions:
            expr = " && ".join(conditions)

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
    results = col.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=[
            "exercise_id", "name", "name_cn", "text",
            "primary_muscles", "equipment", "difficulty",
        ],
    )

    return [
        {
            "exercise_id": hit.entity.get("exercise_id"),
            "name": hit.entity.get("name"),
            "name_cn": hit.entity.get("name_cn"),
            "text": hit.entity.get("text"),
            "primary_muscles": hit.entity.get("primary_muscles"),
            "equipment": hit.entity.get("equipment"),
            "difficulty": hit.entity.get("difficulty"),
            "score": hit.score,
        }
        for hit in results[0]
    ]
