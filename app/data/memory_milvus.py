from pymilvus import FieldSchema, CollectionSchema, DataType
from app.core.llm_factory import create_embeddings
from app.core.milvus_client import milvus_manager
from app.data.mongo_client import get_db
from app.conf.settings import config
from loguru import logger

COLLECTION = (
    config.memory_milvus_collection
    if hasattr(config, "memory_milvus_collection")
    else "session_memory"
)
DIM = config.embedding_dim


class MemoryVectorStore:
    def __init__(self):
        self._embeddings = None
        self._col = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = create_embeddings()
        return self._embeddings

    def _ensure_collection(self):
        if self._col is not None:
            return
        schema = CollectionSchema(
            [
                FieldSchema(
                    name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
                ),
                FieldSchema(
                    name="user_id", dtype=DataType.VARCHAR, max_length=64
                ),
                FieldSchema(
                    name="memory_id", dtype=DataType.VARCHAR, max_length=64
                ),
                FieldSchema(
                    name="text", dtype=DataType.VARCHAR, max_length=2048
                ),
                FieldSchema(
                    name="bucket", dtype=DataType.VARCHAR, max_length=64
                ),
                FieldSchema(name="importance", dtype=DataType.INT64),
                FieldSchema(
                    name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM
                ),
            ]
        )
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._col = milvus_manager.create_collection_if_not_exists(
            COLLECTION, schema, index_params
        )
        logger.info(f"Memory vector store ready: {COLLECTION}")

    @property
    def col(self):
        self._ensure_collection()
        return self._col

    def search(
        self, user_id: str, query: str, top_k: int = 3
    ) -> list[dict]:
        results = []
        try:
            query_vec = self.embeddings.embed_query(query)
            results = self.col.search(
                data=[query_vec],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k,
                expr=f'user_id == "{user_id}"',
                output_fields=["text", "bucket", "importance"],
            )
            return [
                {
                    "text": hit.entity.get("text"),
                    "bucket": hit.entity.get("bucket"),
                    "importance": hit.entity.get("importance"),
                    "score": hit.score,
                }
                for hit in results[0]
            ]
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []

    def archive_summaries(self, user_id: str, force: bool = False):
        col = get_db()["memory_summaries"]
        query = {} if user_id == "all" else {"user_id": user_id}
        summaries = list(
            col.find(query).sort("importance_score", -1).limit(50)
        )

        if not summaries:
            return

        texts = [s["content"] for s in summaries]
        vectors = self.embeddings.embed_documents(texts)

        entities = [
            {
                "user_id": s["user_id"],
                "memory_id": str(s["_id"]),
                "text": s["content"],
                "bucket": s["bucket"],
                "importance": s["importance_score"],
                "vector": vectors[i],
            }
            for i, s in enumerate(summaries)
        ]

        self.col.insert(entities)
        self.col.flush()
        logger.info(f"Archived {len(entities)} memories to Milvus")
