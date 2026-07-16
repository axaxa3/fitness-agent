from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "FitnessAgent"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # DashScope
    dashscope_api_key: str = ""
    llm_model: str = "qwen-flash"
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024

    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "fitness_agent"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000

    # Memory
    memory_window_size: int = 10
    memory_top_k: int = 3
    memory_importance_threshold: int = 5
    memory_dedup_threshold: float = 0.92
    memory_summary_max_segments: int = 3
    memory_cleanup_interval_minutes: int = 30

    # RAG
    rag_top_k: int = 3

    # Chunk
    chunk_max_size: int = 800
    chunk_overlap: int = 100


config = Settings()
