from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.conf.settings import config


def create_llm(temperature: float = 0.7, model: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or config.llm_model,
        api_key=config.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
    )


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=config.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions=config.embedding_dim,
    )
