from pymongo import MongoClient
from app.conf.settings import config
from loguru import logger

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(config.mongo_uri)
        logger.info(f"MongoDB connected: {config.mongo_uri}")
    return _client


def get_db():
    return get_client()[config.mongo_db]


def close_client():
    global _client
    if _client:
        _client.close()
        _client = None
        logger.info("MongoDB connection closed")
