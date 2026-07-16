from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-flash"
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024
    temperature: float = 0.7

    @classmethod
    def from_env(cls) -> "LLMConfig":
        import os
        from dotenv import load_dotenv
        load_dotenv()
        return cls(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            base_url=os.getenv("DASHSCOPE_BASE_URL", cls.base_url),
            model=os.getenv("LLM_MODEL", cls.model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
        )


@dataclass
class MongoConfig:
    uri: str = "mongodb://localhost:27017"
    db_name: str = "fitness_agent"

    @classmethod
    def from_env(cls) -> "MongoConfig":
        import os
        from dotenv import load_dotenv
        load_dotenv()
        return cls(
            uri=os.getenv("MONGO_URI", cls.uri),
            db_name=os.getenv("MONGO_DB", cls.db_name),
        )


@dataclass
class MilvusConfig:
    host: str = "localhost"
    port: int = 19530

    @classmethod
    def from_env(cls) -> "MilvusConfig":
        import os
        from dotenv import load_dotenv
        load_dotenv()
        return cls(
            host=os.getenv("MILVUS_HOST", cls.host),
            port=int(os.getenv("MILVUS_PORT", "19530")),
        )
