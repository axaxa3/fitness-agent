from pymilvus import connections, Collection, utility
from app.conf.settings import config
from loguru import logger


class MilvusManager:
    def __init__(self):
        self._connected = False

    def connect(self):
        if self._connected:
            return
        connections.connect(
            alias="default",
            host=config.milvus_host,
            port=config.milvus_port,
            timeout=config.milvus_timeout // 1000,
        )
        self._connected = True
        logger.info(f"Milvus connected: {config.milvus_host}:{config.milvus_port}")

    def close(self):
        if self._connected:
            connections.disconnect("default")
            self._connected = False
            logger.info("Milvus connection closed")

    def has_collection(self, name: str) -> bool:
        self.connect()
        return utility.has_collection(name)

    def get_collection(self, name: str) -> Collection:
        if not self._connected:
            self.connect()
        return Collection(name)

    def create_collection_if_not_exists(
        self, name: str, schema, index_params
    ) -> Collection:
        self.connect()
        if not self.has_collection(name):
            col = Collection(name, schema=schema)
            col.create_index("vector", index_params)
            col.load()
            logger.info(f"Milvus collection created: {name}")
            return col
        col = Collection(name)
        col.load()
        return col


milvus_manager = MilvusManager()
