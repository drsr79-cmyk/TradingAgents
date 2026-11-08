"""Social screening does not send a TypeSafe key from the analysis process.

The previous HTTP client tests described a request this process no longer makes.
"""

import pytest

from tradingagents.agents import post_screen as typesafe
from tradingagents.dataflows.errors import AnalysisProcessHasNoVendorKeys

CANARY = "CANARY-DATA-KEY-9f3a2c7e1b"


@pytest.mark.unit
def test_system_one_does_not_send_the_key(monkeypatch):
    called = []
    monkeypatch.setenv("TYPESAFE_API_KEY", CANARY)
    monkeypatch.setattr(typesafe.requests, "post", lambda *a, **k: called.append((a, k)))

    with pytest.raises(AnalysisProcessHasNoVendorKeys) as raised:
        typesafe.system_one("s", {"is_urgent": {"type": "noul"}})

    assert CANARY not in str(raised.value)
    assert called == []


@pytest.mark.unit
def test_no_key_no_screen(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert typesafe.jev_screen("NVDA") is None


@pytest.mark.unit
def test_a_set_key_still_does_not_screen_or_leave_the_process(monkeypatch):
    called = []
    monkeypatch.setenv("TYPESAFE_API_KEY", CANARY)
    monkeypatch.setenv("TRADINGAGENTS_PROFILE", "online")
    monkeypatch.setattr(typesafe.requests, "post", lambda *a, **k: called.append(a))

    keep, note = typesafe.jev_screen("NVDA")(["NVDA to 200", "other"])

    assert keep == [True, True]
    assert "does not use TYPESAFE_API_KEY" in note
    assert CANARY not in note
    assert called == []


@pytest.mark.unit
def test_the_sentiment_analyst_hands_the_screen_to_both_social_fetchers(monkeypatch):
    from langchain_core.messages import AIMessage

    from tradingagents.agents.analysts import sentiment_analyst

    screen = object()
    seen = []
    monkeypatch.setattr(sentiment_analyst, "jev_screen", lambda ticker: screen)
    monkeypatch.setattr(sentiment_analyst.get_news, "func", lambda *a: "news")
    for name in ("fetch_stocktwits_messages", "fetch_reddit_posts"):
        monkeypatch.setattr(sentiment_analyst, name, lambda *a, screen=None, **k: seen.append(screen) or "")

    class _LLM:
        def with_structured_output(self, *a, **k):
            raise NotImplementedError

        def invoke(self, messages):
            return AIMessage(content="report")

    node = sentiment_analyst.create_sentiment_analyst(_LLM())
    node({"company_of_interest": "NVDA", "trade_date": "2026-01-09", "messages": []})

    assert seen == [screen, screen]
