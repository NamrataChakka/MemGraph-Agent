"""
Conversational agent backed by the graph memory engine.
"""

from __future__ import annotations

from .memory import MemGraphEngine, LLMClient

SYSTEM_TEMPLATE = """\
You are a helpful personal assistant with a long-term memory system.
You remember past conversations and use them to give better, personalised answers.

{memory_context}

Current conversation:"""


class MemGraphAgent:
    """
    Chat agent that:
      1. Retrieves relevant memories before each reply
      2. Injects them into the system prompt
      3. Stores the new turn as an episodic memory after each reply
    """

    def __init__(self, engine: MemGraphEngine, llm: LLMClient, max_history: int = 20):
        self.engine      = engine
        self.llm         = llm
        self.max_history = max_history
        self._history: list[dict] = []

    def chat(self, user_message: str) -> str:
        memory_context = self.engine.build_context(user_message)
        system_prompt  = SYSTEM_TEMPLATE.format(memory_context=memory_context)

        messages = (
            [{"role": "system", "content": system_prompt}]
            + self._history[-self.max_history:]
            + [{"role": "user", "content": user_message}]
        )

        reply = self.llm.chat(messages)

        # update in-session history
        self._history.append({"role": "user",      "content": user_message})
        self._history.append({"role": "assistant",  "content": reply})

        # persist to graph
        self.engine.remember(user_message, reply)

        return reply

    def reset_session(self):
        """Clear in-memory chat history (graph memory persists)."""
        self._history.clear()
