from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.conf.settings import config
from app.core.logger import setup_logger
from app.core.milvus_client import milvus_manager
from app.data.mongo_client import get_client, close_client
from app.data.memory_manager import memory_manager

# 导入 API 路由
from app.onboarding.api import router as onboard_router
from app.plan.api import router as plan_router
from app.training.api import router as training_router
from app.chat.api import router as chat_router

logger = setup_logger(config.debug)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"{config.app_name} v{config.app_version} 启动中...")
    logger.info(f"监听: http://{config.host}:{config.port}")
    logger.info(f"API 文档: http://{config.host}:{config.port}/docs")

    # 连接 MongoDB
    try:
        get_client()
        logger.info("MongoDB 连接成功")
    except Exception as e:
        logger.warning(f"MongoDB 连接失败: {e}")

    # 连接 Milvus
    try:
        milvus_manager.connect()
        logger.info("Milvus 连接成功")
    except Exception as e:
        logger.warning(f"Milvus 连接失败: {e}")

    # 连接记忆向量存储
    try:
        memory_manager.vector_store.connect()
        logger.info("记忆向量存储就绪")
    except Exception as e:
        logger.warning(f"记忆向量存储连接失败: {e}")

    # 启动记忆维护
    try:
        memory_manager.start_maintenance()
        logger.info("记忆维护任务已启动")
    except Exception as e:
        logger.warning(f"记忆维护启动失败: {e}")

    logger.info("=" * 60)

    yield

    # 关闭
    try:
        memory_manager.stop_maintenance()
    except Exception:
        pass
    milvus_manager.close()
    close_client()
    logger.info(f"{config.app_name} 关闭")


app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="AI 健身训练助手",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(onboard_router, tags=["引导对话"])
app.include_router(plan_router, tags=["训练计划"])
app.include_router(training_router, tags=["训练执行"])
app.include_router(chat_router, tags=["自由问答"])

# 额外路由：exercise search + profile + history
from fastapi import Query


@app.get("/api/exercise/search")
async def search_exercises(
    q: str = Query(..., description="搜索关键词"),
    equipment: str | None = None,
    difficulty: int | None = None,
):
    from app.tools.exercise_search import search_exercises_semantic
    filters = {}
    if equipment:
        filters["equipment"] = equipment
    if difficulty:
        filters["difficulty"] = difficulty
    results = search_exercises_semantic(q, top_k=10, filters=filters or None)
    return {"results": results}


@app.get("/api/profile")
async def get_profile(user_id: str):
    from app.data.user_profile import get_profile as gp
    p = gp(user_id)
    if not p:
        return {"error": "user not found"}
    p.pop("_id", None)
    return p


@app.get("/api/history")
async def get_history(user_id: str, limit: int = 30, skip: int = 0):
    from app.data.training_log import get_training_logs
    logs = get_training_logs(user_id, limit=limit, skip=skip)
    for log in logs:
        log["_id"] = str(log["_id"])
    return {"logs": logs}


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": config.app_name, "version": config.app_version}


# 静态文件
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": f"Welcome to {config.app_name}", "version": config.app_version, "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.host, port=config.port, reload=config.debug)
