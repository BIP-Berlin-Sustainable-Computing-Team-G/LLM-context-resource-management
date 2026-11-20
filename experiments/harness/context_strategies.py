"""Context Management Strategies for Multi-Turn LLM Conversations.

Answers RQ2 (notes/research-questions.md, notes/implementation.md):
How do context strategies (full history, sliding window, summarised history)
compare in energy and answer quality?

Strategies:
1. FullHistoryStrategy:
   Prompt = system + all historical turns + current user query.
   Prefix stable? YES (append-only). Extra LLM calls: NONE.

2. SlidingWindowStrategy:
   Prompt = system + last k turns + current user query (k = 5 by default).
   Prefix stable? NO (start token shifts every turn). Extra LLM calls: NONE.

3. BatchSummaryStrategy:
   Prompt = system + summary + last k turns + current user query.
   Re-summarises older turns when exceeding a token budget.
   Prefix stable? YES between re-summaries; NO at re-summary boundaries.
   Extra LLM calls: ONE per batch re-summary.

4. RollingSummaryStrategy:
   Prompt = system + summary + last turn + current user query.
   Summary updated on every single turn.
   Prefix stable? NO (summary text rewritten every turn).
   Extra LLM calls: ONE per turn.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ContextStrategy:
    """Base class for dialogue context management strategies."""

    def format_request(
        self,
        system_prompt: str,
        history: List[Message],
        current_query: str,
        summarizer_fn: Optional[Callable[[List[Message]], str]] = None,
    ) -> List[Message]:
        """Produce the OpenAI-compatible message list sent to llama-server."""
        raise NotImplementedError


class FullHistoryStrategy(ContextStrategy):
    """Full verbatim dialogue history.

    Re-sends all past messages. The prefix is identical and append-only,
    maximizing prompt cache reuse at the cost of unbounded VRAM.
    """

    def format_request(
        self,
        system_prompt: str,
        history: List[Message],
        current_query: str,
        summarizer_fn: Optional[Callable[[List[Message]], str]] = None,
    ) -> List[Message]:
        messages = [Message(role="system", content=system_prompt)]
        messages.extend(history)
        messages.append(Message(role="user", content=current_query))
        return messages


class SlidingWindowStrategy(ContextStrategy):
    """Strict sliding window of the last k dialogue turns (default k = 5).

    One turn = (user message, assistant reply) pair, so k turns = 2 * k messages.
    Older turns are discarded. Binds memory, but shifts prompt start on every turn,
    causing a near-total prompt cache miss (only the system prompt matches).
    """

    def __init__(self, k_turns: int = 5):
        self.k_turns = k_turns

    def format_request(
        self,
        system_prompt: str,
        history: List[Message],
        current_query: str,
        summarizer_fn: Optional[Callable[[List[Message]], str]] = None,
    ) -> List[Message]:
        messages = [Message(role="system", content=system_prompt)]
        max_messages = self.k_turns * 2
        window_slice = history[-max_messages:] if len(history) > max_messages else history
        messages.extend(window_slice)
        messages.append(Message(role="user", content=current_query))
        return messages


class BatchSummaryStrategy(ContextStrategy):
    """Batch summarisation triggered when history exceeds a token or turn budget.

    Maintains a stable summary of older dialogue plus the last k turns in full.
    The prefix remains stable between batch re-summaries.
    """

    def __init__(self, k_recent_turns: int = 5, trigger_turns: int = 10):
        self.k_recent_turns = k_recent_turns
        self.trigger_turns = trigger_turns
        self.cached_summary: Optional[str] = None
        self.summarized_turn_count: int = 0

    def format_request(
        self,
        system_prompt: str,
        history: List[Message],
        current_query: str,
        summarizer_fn: Optional[Callable[[List[Message]], str]] = None,
    ) -> List[Message]:
        total_turns = len(history) // 2
        recent_msg_count = self.k_recent_turns * 2

        # Check if we should perform / update a batch summary
        if total_turns >= self.trigger_turns and summarizer_fn is not None:
            older_history = history[:-recent_msg_count] if total_turns > self.k_recent_turns else []
            if older_history and (len(older_history) // 2 > self.summarized_turn_count):
                self.cached_summary = summarizer_fn(older_history)
                self.summarized_turn_count = len(older_history) // 2

        messages = [Message(role="system", content=system_prompt)]

        if self.cached_summary:
            summary_msg = (
                f"[Summary of prior conversation up to turn {self.summarized_turn_count}]: "
                f"{self.cached_summary}"
            )
            messages.append(Message(role="system", content=summary_msg))
            messages.extend(history[-recent_msg_count:])
        else:
            messages.extend(history)

        messages.append(Message(role="user", content=current_query))
        return messages


class RollingSummaryStrategy(ContextStrategy):
    """Incremental rolling summary updated on every turn.

    Appends the latest dialogue turn into an evolving running summary.
    Only the summary and the most recent turn are preserved in the prompt.
    Hostile to prompt cache because the summary text changes every turn.
    """

    def __init__(self):
        self.current_summary: Optional[str] = None
        self.last_summarized_turn: int = 0

    def format_request(
        self,
        system_prompt: str,
        history: List[Message],
        current_query: str,
        summarizer_fn: Optional[Callable[[str, Message, Message], str]] = None,
    ) -> List[Message]:
        total_turns = len(history) // 2

        # Update rolling summary if a new turn was added
        if total_turns > self.last_summarized_turn and summarizer_fn is not None:
            last_user = history[-2]
            last_assistant = history[-1]
            prev_sum = self.current_summary or "Beginning of conversation."
            self.current_summary = summarizer_fn(prev_sum, last_user, last_assistant)
            self.last_summarized_turn = total_turns

        messages = [Message(role="system", content=system_prompt)]

        if self.current_summary:
            summary_msg = (
                f"[Rolling summary of conversation history up to turn {self.last_summarized_turn}]: "
                f"{self.current_summary}"
            )
            messages.append(Message(role="system", content=summary_msg))
            # Include the last turn if available
            if len(history) >= 2:
                messages.extend(history[-2:])
        else:
            messages.extend(history)

        messages.append(Message(role="user", content=current_query))
        return messages
