import queue
from typing import Any
from loguru import logger

_sessions: dict[str, queue.Queue] = {}


class SessionManager:
    @staticmethod
    def create(session_id: str) -> queue.Queue:
        q: queue.Queue[Any] = queue.Queue()
        _sessions[session_id] = q
        logger.debug(f"SSE session created: {session_id}")
        return q

    @staticmethod
    def get(session_id: str) -> queue.Queue | None:
        return _sessions.get(session_id)

    @staticmethod
    def remove(session_id: str):
        q = _sessions.pop(session_id, None)
        if q:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            logger.debug(f"SSE session removed: {session_id}")

    @staticmethod
    def exists(session_id: str) -> bool:
        return session_id in _sessions


def push_to_session(session_id: str, event_type: str, data: Any):
    q = SessionManager.get(session_id)
    if q:
        q.put({"type": event_type, "data": data})


def push_progress(session_id: str, message: str):
    push_to_session(session_id, "progress", {"message": message})


def push_delta(session_id: str, content: str):
    push_to_session(session_id, "delta", {"content": content})


def push_final(session_id: str, result: dict):
    push_to_session(session_id, "final", result)


def push_error(session_id: str, error: str):
    push_to_session(session_id, "error", {"message": error})
