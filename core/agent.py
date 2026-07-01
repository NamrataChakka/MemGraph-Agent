"""
Conversational agent backed by the graph memory engine.
"""

from __future__ import annotations

from datetime import date
from typing import Iterator

from .memory import MemGraphEngine, LLMClient

SYSTEM_TEMPLATE = """\
You are a helpful personal assistant with a long-term memory system.
You remember past conversations and use them to give better, personalised answers.
Keep responses concise — 50 words or fewer unless the user explicitly asks for more detail, a list, or an explanation.
Today's date is {today}.

CRITICAL RULES:
- You MUST use tool calls to perform actions. NEVER pretend you did something without calling a tool.
- To add a book, you MUST call book_add. To add list items, you MUST call list_add. Etc.
- When using tools, ONLY include information the user explicitly provided. Never guess or fabricate dates, names, or details.
- If a past memory says something failed, try again anyway — the issue may be fixed.
{memory_section}
Current conversation:"""


class MemGraphAgent:
    """
    Chat agent that:
      1. Retrieves relevant memories before each reply
      2. Injects them into the system prompt
      3. Stores the new turn as an episodic memory after each reply
    """

    def __init__(self, engine: MemGraphEngine, llm: LLMClient, max_history: int = 5):
        self.engine      = engine
        self.llm         = llm
        self.max_history = max_history
        self._history: list[dict] = []

    _SKIP_MEMORY_TOOLS = {
        "list_view", "list_add", "list_remove", "list_clear", "lists_all",
        "note_write", "note_read", "note_list", "note_delete",
        "event_add", "events_list", "event_delete",
        "expense_add", "expenses_summary",
        "book_add", "book_list", "book_update", "book_delete",
        "weather_get",
    }

    def chat(self, user_message: str, tools: list[dict] | None = None) -> tuple[str, bool]:
        """Chat with the user. Returns (reply, should_remember)."""
        memory_context = self.engine.build_context(user_message)
        memory_section = f"\n{memory_context}" if memory_context else ""
        system_prompt  = SYSTEM_TEMPLATE.format(today=date.today().isoformat(), memory_section=memory_section)

        messages = (
            [{"role": "system", "content": system_prompt}]
            + self._history[-(self.max_history * 2):]
            + [{"role": "user", "content": user_message}]
        )

        tools_called: list[str] = []
        if tools:
            reply, tools_called = self.llm.chat_with_tools(messages, tools)
        else:
            reply = self.llm.chat(messages)

        self._history.append({"role": "user",      "content": user_message})
        self._history.append({"role": "assistant",  "content": reply})

        should_remember = True
        if tools_called and all(t in self._SKIP_MEMORY_TOOLS for t in tools_called):
            should_remember = False

        return reply, should_remember

    def stream_chat(self, user_message: str) -> Iterator[str]:
        """Stream a reply token by token. Caller must invoke remember_turn() separately."""
        memory_context = self.engine.build_context(user_message)
        memory_section = f"\n{memory_context}" if memory_context else ""
        system_prompt  = SYSTEM_TEMPLATE.format(today=date.today().isoformat(), memory_section=memory_section)

        messages = (
            [{"role": "system", "content": system_prompt}]
            + self._history[-(self.max_history * 2):]
            + [{"role": "user", "content": user_message}]
        )

        full_reply = ""
        for chunk in self.llm.stream_chat(messages):
            full_reply += chunk
            yield chunk

        self._history.append({"role": "user",      "content": user_message})
        self._history.append({"role": "assistant",  "content": full_reply})

    def remember_turn(self, user_message: str, reply: str) -> None:
        """Persist a conversation turn to graph memory (call after reply sent)."""
        self.engine.remember(user_message, reply)

    def reset_session(self):
        """Clear in-memory chat history (graph memory persists)."""
        self._history.clear()
