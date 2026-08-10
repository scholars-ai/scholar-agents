"""采集与清洗测试：字段提取、HTML 清洗、去重键、role/full_text 行为。

不打网络：feed 用本地 XML 字符串，页面抓取被 mock。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scholar_agents.sourcing import fetcher

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
<channel>
  <title>Test Feed</title>
  <item>
    <title><![CDATA[千问开放平台上线]]></title>
    <link>https://example.com/a</link>
    <description><![CDATA[<p>平台今日上线，覆盖物流、房产等十多个领域。</p>]]></description>
    <pubDate>Mon, 10 Aug 2026 02:07:18 GMT</pubDate>
    <guid isPermaLink="false">abc123</guid>
    <author>noreply@example.com (公众号：某某)</author>
  </item>
  <item>
    <title>No body here</title>
    <link>https://example.com/b</link>
    <guid>def456</guid>
  </item>
</channel>
</rss>
"""


def _mock_get(content: bytes | str) -> MagicMock:
    resp = MagicMock()
    resp.content = content if isinstance(content, bytes) else content.encode()
    resp.text = content.decode() if isinstance(content, bytes) else content
    resp.raise_for_status = MagicMock()
    return resp


class TestFetchFeed:
    def test_parses_entries(self) -> None:
        with patch("httpx.get", return_value=_mock_get(RSS)):
            entries = fetcher.fetch_feed("https://example.com/feed.xml")
        assert len(entries) == 2
        assert entries[0]["title"] == "千问开放平台上线"

    def test_empty_feed_raises(self) -> None:
        empty = '<?xml version="1.0"?><rss version="2.0"><channel><title>x</title></channel></rss>'
        with patch("httpx.get", return_value=_mock_get(empty)), \
             pytest.raises(fetcher.FetchError, match="no entries"):
            fetcher.fetch_feed("https://example.com/feed.xml")

    def test_network_error_raises_fetch_error(self) -> None:
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("boom")), \
             pytest.raises(fetcher.FetchError, match="feed request failed"):
            fetcher.fetch_feed("https://example.com/feed.xml")


class TestBuildItem:
    def _entries(self) -> list[dict]:
        with patch("httpx.get", return_value=_mock_get(RSS)):
            return fetcher.fetch_feed("https://example.com/feed.xml")

    def test_signal_source_keeps_summary_without_page_fetch(self) -> None:
        entry = self._entries()[0]
        with patch.object(fetcher, "fetch_page_text") as page:
            item = fetcher.build_item(entry, role="signal", full_text="rss_description")
        page.assert_not_called(), "signal 源不该抓页面"
        assert item is not None
        assert "平台今日上线" in item.content
        assert "<p>" not in item.content, "HTML 标签必须清掉"

    def test_material_source_fetches_page_when_configured(self) -> None:
        entry = self._entries()[0]
        long_body = "完整正文。" * 200
        with patch.object(fetcher, "fetch_page_text", return_value=long_body) as page:
            item = fetcher.build_item(entry, role="material", full_text="fetch_page")
        page.assert_called_once()
        assert item is not None
        assert len(item.content) > 500, "抓到的原文更长时应替换摘要"

    def test_material_falls_back_to_summary_when_page_fails(self) -> None:
        entry = self._entries()[0]
        with patch.object(fetcher, "fetch_page_text", return_value=""):
            item = fetcher.build_item(entry, role="material", full_text="fetch_page")
        assert item is not None
        assert "平台今日上线" in item.content, "抓页面失败要降级用摘要，而不是丢掉该条"

    def test_entry_without_body_is_skipped(self) -> None:
        entry = self._entries()[1]
        with patch.object(fetcher, "fetch_page_text", return_value=""):
            assert fetcher.build_item(entry, role="signal", full_text="rss_description") is None

    def test_guid_preferred_as_dedup_key(self) -> None:
        entry = self._entries()[0]
        item = fetcher.build_item(entry, role="signal", full_text="rss_description")
        assert item is not None
        assert item.content_hash.startswith("guid:"), "有 guid 时应优先用它做去重键"

    def test_falls_back_to_body_hash_without_guid(self) -> None:
        entry = {"title": "T", "link": "https://x.com", "description": "body text here " * 20}
        item = fetcher.build_item(entry, role="signal", full_text="rss_description")
        assert item is not None
        assert item.content_hash.startswith("body:")

    def test_published_at_parsed_as_utc(self) -> None:
        entry = self._entries()[0]
        item = fetcher.build_item(entry, role="signal", full_text="rss_description")
        assert item is not None and item.published_at is not None
        assert item.published_at.year == 2026 and item.published_at.month == 8

    def test_content_truncated_to_limit(self) -> None:
        entry = {"title": "T", "description": "长" * 50_000}
        item = fetcher.build_item(entry, role="signal", full_text="rss_description")
        assert item is not None
        assert len(item.content) == fetcher.MAX_CONTENT_CHARS


class TestCleanHtml:
    def test_strips_tags_and_entities(self) -> None:
        out = fetcher._clean_html("<p>a&nbsp;b &amp; c</p><br/><div>d</div>")
        assert "<" not in out and "&nbsp;" not in out
        assert "a b & c" in out

    def test_empty_input(self) -> None:
        assert fetcher._clean_html("") == ""


class TestFetchPageText:
    def test_returns_empty_on_http_error(self) -> None:
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("nope")):
            assert fetcher.fetch_page_text("https://example.com/x") == ""


class TestMaxItemsCap:
    """arXiv 类源单次返回数百条（实测 cs.AI 295 篇 / cs.CL 119 篇），必须限量。"""

    def test_default_cap_is_conservative(self) -> None:
        from scholar_agents.sourcing import handler
        assert handler.DEFAULT_MAX_ITEMS == 30

    def test_config_overrides_cap(self) -> None:
        from scholar_agents.sourcing.handler import _source_config
        cfg = {"role": "material", "full_text": "rss_description", "max_items": 15}
        assert _source_config({"fetch_config": cfg}) == ("material", "rss_description", 15)

    def test_missing_config_uses_safe_defaults(self) -> None:
        from scholar_agents.sourcing.handler import DEFAULT_MAX_ITEMS, _source_config
        assert _source_config({}) == ("signal", "rss_description", DEFAULT_MAX_ITEMS)

    def test_bad_cap_falls_back_to_default(self) -> None:
        from scholar_agents.sourcing.handler import DEFAULT_MAX_ITEMS, _source_config
        _, _, cap = _source_config({"fetch_config": {"max_items": "not-a-number"}})
        assert cap == DEFAULT_MAX_ITEMS

    def test_cap_is_at_least_one(self) -> None:
        from scholar_agents.sourcing.handler import _source_config
        _, _, cap = _source_config({"fetch_config": {"max_items": 0}})
        assert cap == 1
