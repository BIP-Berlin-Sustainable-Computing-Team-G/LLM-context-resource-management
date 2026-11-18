"""Unit tests for Context Management Strategies (RQ2).

Run via: python3 -m unittest experiments/harness/test_context_strategies.py
"""

import unittest
from experiments.harness.context_strategies import (
    Message,
    FullHistoryStrategy,
    SlidingWindowStrategy,
    BatchSummaryStrategy,
    RollingSummaryStrategy,
)


def make_turn(turn_idx: int):
    return [
        Message(role="user", content=f"User question {turn_idx}"),
        Message(role="assistant", content=f"Assistant reply {turn_idx}"),
    ]


class TestContextStrategies(unittest.TestCase):
    def setUp(self):
        self.sys = "You are a helpful assistant."
        self.query = "What is the capital of France?"

    def test_full_history(self):
        strategy = FullHistoryStrategy()
        history = []
        for t in range(1, 10):
            history.extend(make_turn(t))

        req = strategy.format_request(self.sys, history, self.query)
        self.assertEqual(req[0].role, "system")
        self.assertEqual(req[0].content, self.sys)
        # 1 sys + 18 history messages (9 turns) + 1 current user query = 20 messages
        self.assertEqual(len(req), 20)
        self.assertEqual(req[1].content, "User question 1")
        self.assertEqual(req[-1].content, self.query)

    def test_sliding_window(self):
        strategy = SlidingWindowStrategy(k_turns=5)
        history = []
        for t in range(1, 11):  # 10 turns
            history.extend(make_turn(t))

        req = strategy.format_request(self.sys, history, self.query)
        # 1 sys + 10 msgs (last 5 turns: turns 6, 7, 8, 9, 10) + 1 query = 12 messages
        self.assertEqual(len(req), 12)
        self.assertEqual(req[1].content, "User question 6")
        self.assertEqual(req[-2].content, "Assistant reply 10")
        self.assertEqual(req[-1].content, self.query)

    def test_sliding_window_short_history(self):
        strategy = SlidingWindowStrategy(k_turns=5)
        history = []
        for t in range(1, 4):  # only 3 turns
            history.extend(make_turn(t))

        req = strategy.format_request(self.sys, history, self.query)
        # 1 sys + 6 msgs (3 turns) + 1 query = 8 messages
        self.assertEqual(len(req), 8)
        self.assertEqual(req[1].content, "User question 1")

    def test_batch_summary(self):
        strategy = BatchSummaryStrategy(k_recent_turns=3, trigger_turns=6)
        history = []
        for t in range(1, 8):  # 7 turns total
            history.extend(make_turn(t))

        mock_summarizer = lambda msgs: f"Condensed {len(msgs)//2} older turns."

        req = strategy.format_request(self.sys, history, self.query, mock_summarizer)
        # Turns 1..4 (4 turns) are summarized; Turns 5..7 (3 turns = 6 msgs) are kept
        self.assertEqual(req[0].role, "system")
        self.assertTrue("Condensed 4 older turns" in req[1].content)
        self.assertEqual(req[2].content, "User question 5")
        self.assertEqual(req[-1].content, self.query)

    def test_rolling_summary(self):
        strategy = RollingSummaryStrategy()
        history = []
        
        def mock_rolling(prev, user, assist):
            return f"{prev} -> ({user.content} : {assist.content})"

        for t in range(1, 4):  # 3 turns
            history.extend(make_turn(t))
            req = strategy.format_request(self.sys, history, self.query, mock_rolling)

        self.assertEqual(req[0].role, "system")
        self.assertTrue("Rolling summary" in req[1].content)
        self.assertTrue("User question 3" in req[1].content)
        # Includes summary + last turn (turn 3) + query
        self.assertEqual(req[-3].content, "User question 3")
        self.assertEqual(req[-2].content, "Assistant reply 3")
        self.assertEqual(req[-1].content, self.query)


if __name__ == "__main__":
    unittest.main()
