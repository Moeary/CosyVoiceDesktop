"""可暂停、可续跑的分角会话：LangGraph checkpointer + interrupt。

- 每篇文本（连同 Skill 哈希）对应一个 thread_id，状态逐步写入 SQLite 检查点。
- 校对结束后在 roster_review 暂停，交出角色名单；用户确认或提出意见后用 resume() 继续。
- 某一步失败（超时、限流）时，用 retry() 从最近的检查点续跑，已完成的批次不会重做；
  程序重启后也能接着上次的进度。
"""

import hashlib
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .llm import PROJECT_ROOT

CHECKPOINT_PATH = PROJECT_ROOT / 'cache' / 'role_agent_checkpoints.sqlite'
RECURSION_LIMIT = 400


def open_checkpointer(path: Optional[str] = None):
    """SQLite 检查点；连接会在多个工作线程间先后使用，所以关闭同线程检查。"""
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = Path(path) if path else CHECKPOINT_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(db_path), check_same_thread=False))


def thread_id_for(document_text: str, skill_digest: str) -> str:
    return hashlib.sha256(f"{skill_digest}\n{document_text}".encode('utf-8')).hexdigest()[:16]


class RoleAgentSession:
    """包装 RoleAgent 的一次分角会话。status：new / running（中途失败）/ review（等用户确认）/ done。"""

    def __init__(self, agent, checkpointer=None, thread_id: Optional[str] = None):
        agent.interactive = True
        self.agent = agent
        self.checkpointer = checkpointer or InMemorySaver()
        self.app = agent.build_graph(self.checkpointer)
        self.thread_id = thread_id or uuid.uuid4().hex[:16]
        self.config = {'configurable': {'thread_id': self.thread_id}, 'recursion_limit': RECURSION_LIMIT}

    def snapshot(self):
        return self.app.get_state(self.config)

    def status(self) -> str:
        snapshot = self.snapshot()
        if not snapshot.values:
            return 'new'
        if snapshot.interrupts:
            return 'review'
        return 'running' if snapshot.next else 'done'

    @property
    def state(self) -> dict:
        return self.snapshot().values

    def pending(self) -> Optional[dict]:
        """等待用户确认时的角色名单（interrupt 负载），否则为 None。"""
        interrupts = self.snapshot().interrupts
        return interrupts[0].value if interrupts else None

    def reset(self):
        self.checkpointer.delete_thread(self.thread_id)

    def start(self, document_text: str, voice_configs=None, fallback_speaker: str = '') -> Optional[dict]:
        """从头开始；返回待确认的名单，跑完则返回 None。"""
        if self.status() != 'new':
            self.reset()
        self.app.invoke({
            'document_text': document_text,
            'voice_configs': list(voice_configs or []),
            'fallback_speaker': fallback_speaker,
            'trace': [],
        }, self.config)
        return self.pending()

    def resume(self, decision: dict) -> Optional[dict]:
        """decision：{'action': 'approve'|'revise', 'feedback': 意见原文, 'edits': [...]}"""
        self.app.invoke(Command(resume=decision), self.config)
        return self.pending()

    def retry(self) -> Optional[dict]:
        """从最近的检查点续跑（上一步失败后调用）。"""
        self.app.invoke(None, self.config)
        return self.pending()
