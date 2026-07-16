import json
import time
from dataclasses import dataclass, field
from threading import Thread, Event

from loguru import logger

from app.conf.settings import config
from app.core.llm_factory import create_llm
from app.core.prompt_loader import load_prompt_or_default
from app.data.memory_milvus import MemoryVectorStore
from app.data.mongo_client import get_db

MEMORY_BUCKETS = [
    "training_feedback",
    "injury_attention",
    "user_preferences",
    "plan_iterations",
    "qa_patterns",
]


@dataclass
class MemoryManager:
    vector_store: MemoryVectorStore = field(default_factory=MemoryVectorStore)
    _maintenance_thread: Thread | None = None
    _stop_event: Event = field(default_factory=Event)

    # --- Layer 1: 短期记忆（内存） ---
    @staticmethod
    def build_short_term_context(
        recent_messages: list[dict],
        user_profile: dict | None,
        current_scenario: str = "chat",
        window_size: int = 10,
    ) -> str:
        """构建注入 prompt 的短期上下文"""
        parts = []

        if user_profile:
            fitness = user_profile.get("fitness_profile", {})
            injury = user_profile.get("injury_flags", [])
            parts.append(
                f"[用户画像] 目标:{fitness.get('goal', '')} 水平:{fitness.get('level', '')} "
                f"器械:{', '.join(fitness.get('equipment', []))} 频率:每周{fitness.get('days_per_week', 0)}天"
            )
            if injury:
                active_injuries = [
                    f
                    for f in injury
                    if f.get("status") in ("monitoring", "active")
                ]
                if active_injuries:
                    parts.append(
                        f"[伤病关注] {', '.join([i['area'] for i in active_injuries])}"
                    )

        recent = recent_messages[-window_size:]
        if recent:
            parts.append(
                "[最近对话]\n"
                + "\n".join(
                    [
                        f"{m['role']}: {str(m['content'])[:200]}"
                        for m in recent
                    ]
                )
            )

        return "\n\n".join(parts)

    # --- Layer 2: 中期记忆（MongoDB 主题桶） ---
    def get_bucket_memories(
        self, user_id: str, buckets: list[str] | None = None
    ) -> dict[str, list]:
        """按桶获取中期记忆"""
        if buckets is None:
            buckets = MEMORY_BUCKETS
        col = get_db()["memory_summaries"]
        result = {}
        for bucket in buckets:
            docs = list(
                col.find({"user_id": user_id, "bucket": bucket})
                .sort("importance_score", -1)
                .limit(config.memory_summary_max_segments)
            )
            result[bucket] = [d["content"] for d in docs]
        return result

    def save_to_bucket(
        self, user_id: str, bucket: str, content: str, importance: int
    ):
        col = get_db()["memory_summaries"]
        col.insert_one(
            {
                "user_id": user_id,
                "bucket": bucket,
                "content": content,
                "importance_score": importance,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

    # --- Layer 3: 长期记忆（Milvus session_memory） ---
    def archive_to_long_term(self, user_id: str, force: bool = False):
        """将中期记忆摘要归档到 Milvus 长期存储"""
        self.vector_store.archive_summaries(user_id, force)

    # --- Layer 4: 用户画像更新 ---
    @staticmethod
    def update_user_profile(user_id: str, scoped_updates: dict):
        """根据训练反馈更新用户画像的特定字段"""
        from app.data.user_profile import update_profile

        update_profile(user_id, scoped_updates)

    # --- 综合注入 ---
    def inject_memories(
        self,
        user_id: str,
        scenario: str,
        current_query: str,
    ) -> str:
        """根据场景选择性注入记忆"""
        parts = []

        # 中期记忆（按场景选择性注入）
        scenario_buckets = {
            "training": ["training_feedback", "injury_attention"],
            "plan": ["plan_iterations", "user_preferences", "injury_attention"],
            "chat": ["qa_patterns", "user_preferences"],
            "onboarding": [],
        }
        buckets = scenario_buckets.get(
            scenario, ["training_feedback", "user_preferences"]
        )
        bucket_memories = self.get_bucket_memories(user_id, buckets)

        for bucket, contents in bucket_memories.items():
            if contents:
                bucket_cn = {
                    "training_feedback": "训练反馈",
                    "injury_attention": "伤病关注",
                    "user_preferences": "用户偏好",
                    "plan_iterations": "计划调整",
                    "qa_patterns": "常见问题",
                }
                parts.append(
                    f"[{bucket_cn.get(bucket, bucket)}]\n"
                    + "\n".join([f"- {c}" for c in contents])
                )

        # 长期记忆（语义检索）
        long_term = []
        try:
            long_term = self.vector_store.search(
                user_id, current_query, top_k=3
            )
            if long_term:
                parts.append(
                    "[历史相关记忆]\n"
                    + "\n".join([f"- {m['text']}" for m in long_term])
                )
        except Exception as e:
            logger.debug(f"Long-term memory search skipped: {e}")

        return "\n\n".join(parts)

    # --- 记忆归约 ---
    def summarize_and_store(
        self,
        user_id: str,
        messages: list[dict],
        existing_memories: list[str],
    ) -> list[dict]:
        """对话达到阈值后，触发 LLM 摘要生成并存储到中期桶"""
        prompt_template = load_prompt_or_default(
            "memory_summary",
            "对话: {conversation}\n已有: {existing_memories}\n返回JSON记忆数组。",
        )
        conversation_text = json.dumps(messages[-20:], ensure_ascii=False)
        prompt = prompt_template.replace("{conversation}", conversation_text)
        prompt = prompt.replace(
            "{existing_memories}",
            json.dumps(existing_memories, ensure_ascii=False),
        )

        memories = []
        try:
            llm = create_llm(temperature=0.3)
            response = llm.invoke([{"role": "user", "content": prompt}])
            content = response.content
            if isinstance(content, str):
                memories = json.loads(content)
            else:
                memories = json.loads(str(content))

            for mem in memories:
                if (
                    mem.get("importance", 0)
                    >= config.memory_importance_threshold
                ):
                    self.save_to_bucket(
                        user_id,
                        mem["bucket"],
                        mem["content"],
                        mem["importance"],
                    )
            return memories
        except Exception as e:
            logger.error(f"Memory summarization failed: {e}")
            return []

    # --- 后台维护 ---
    def start_maintenance(self):
        def _run():
            while not self._stop_event.is_set():
                self._stop_event.wait(
                    config.memory_cleanup_interval_minutes * 60
                )
                if self._stop_event.is_set():
                    break
                try:
                    self.archive_to_long_term("all")
                except Exception as e:
                    logger.warning(f"Memory maintenance error: {e}")

        self._maintenance_thread = Thread(target=_run, daemon=True)
        self._maintenance_thread.start()
        logger.info("Memory maintenance started")

    def stop_maintenance(self):
        self._stop_event.set()
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5)


# 全局实例
memory_manager = MemoryManager()
