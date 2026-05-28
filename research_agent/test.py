import pytest
import json
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────
# tools.py tests
# ─────────────────────────────────────────────

class TestWebSearch:
    @patch("tools.DDGS")
    def test_returns_json(self, mock_ddgs):
        """web_search should return a valid JSON string."""
        from tools import web_search
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "Test", "href": "https://example.com", "body": "Some snippet"}
        ]
        result = web_search("test query")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["title"] == "Test"
        assert parsed[0]["url"] == "https://example.com"

    @patch("tools.DDGS")
    def test_handles_exception(self, mock_ddgs):
        """web_search should return error string on failure."""
        from tools import web_search
        mock_ddgs.return_value.__enter__.side_effect = Exception("Network error")
        result = web_search("test query")
        assert "Search failed" in result

    @patch("tools.DDGS")
    def test_respects_max_results(self, mock_ddgs):
        """web_search should pass max_results correctly."""
        from tools import web_search
        mock_instance = mock_ddgs.return_value.__enter__.return_value
        mock_instance.text.return_value = []
        web_search("query", max_results=3)
        mock_instance.text.assert_called_once_with("query", max_results=3)


class TestReadWebpage:
    @patch("tools.requests.get")
    def test_returns_text(self, mock_get):
        """read_webpage should return cleaned text from a page."""
        from tools import read_webpage
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello world</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        result = read_webpage("https://example.com")
        assert "Hello world" in result

    @patch("tools.requests.get")
    def test_truncates_long_content(self, mock_get):
        """read_webpage should truncate content over 4000 chars."""
        from tools import read_webpage
        mock_response = MagicMock()
        mock_response.text = f"<html><body><p>{'x' * 5000}</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        result = read_webpage("https://example.com")
        assert len(result) <= 4000

    @patch("tools.requests.get")
    def test_handles_exception(self, mock_get):
        """read_webpage should return error string on failure."""
        from tools import read_webpage
        mock_get.side_effect = Exception("Timeout")
        result = read_webpage("https://example.com")
        assert "Could not read page" in result

    @patch("tools.requests.get")
    def test_strips_scripts_and_styles(self, mock_get):
        """read_webpage should remove script and style tags."""
        from tools import read_webpage
        mock_response = MagicMock()
        mock_response.text = """
            <html><body>
            <script>alert('bad')</script>
            <style>.hidden{display:none}</style>
            <p>Good content</p>
            </body></html>
        """
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        result = read_webpage("https://example.com")
        assert "alert" not in result
        assert "display:none" not in result
        assert "Good content" in result


# ─────────────────────────────────────────────
# agent.py tests
# ─────────────────────────────────────────────

class TestExtractSources:
    def test_extracts_markdown_links(self):
        """extract_sources should parse markdown links from Sources section."""
        from agent import extract_sources
        report = """
## Summary
Some summary here.

## Key Findings
- Finding 1

## Sources
- [CNN Article](https://cnn.com/article)
- [BBC News](https://bbc.com/news)
        """
        sources = extract_sources(report)
        assert len(sources) == 2
        assert sources[0]["title"] == "CNN Article"
        assert sources[0]["url"] == "https://cnn.com/article"
        assert sources[1]["title"] == "BBC News"

    def test_returns_empty_if_no_sources(self):
        """extract_sources should return empty list when no Sources section."""
        from agent import extract_sources
        report = "## Summary\nNo sources here."
        sources = extract_sources(report)
        assert sources == []

    def test_ignores_malformed_links(self):
        """extract_sources should skip lines that aren't markdown links."""
        from agent import extract_sources
        report = """
## Sources
- Not a link
- [Valid](https://example.com)
- also not a link
        """
        sources = extract_sources(report)
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com"


class TestBuildSystemPrompt:
    def test_neutral_prompt_contains_balanced(self):
        """NEUTRAL bias should mention balanced/unbiased language."""
        from agent import build_system_prompt
        prompt = build_system_prompt("NEUTRAL")
        assert "NEUTRAL" in prompt or "balanced" in prompt.lower()

    def test_left_prompt_contains_progressive(self):
        """LEFT bias prompt should contain progressive framing instructions."""
        from agent import build_system_prompt
        prompt = build_system_prompt("LEFT")
        assert "PROGRESSIVE" in prompt or "progressive" in prompt.lower()

    def test_right_prompt_contains_conservative(self):
        """RIGHT bias prompt should contain conservative framing instructions."""
        from agent import build_system_prompt
        prompt = build_system_prompt("RIGHT")
        assert "CONSERVATIVE" in prompt or "conservative" in prompt.lower()

    def test_unknown_bias_defaults_to_neutral(self):
        """Unknown bias preference should fall back to NEUTRAL."""
        from agent import build_system_prompt
        prompt = build_system_prompt("UNKNOWN_VALUE")
        neutral_prompt = build_system_prompt("NEUTRAL")
        assert prompt == neutral_prompt


# ─────────────────────────────────────────────
# bias_agent.py tests
# ─────────────────────────────────────────────

class TestIsControversial:
    @patch("bias_agent.client")
    def test_returns_true_for_yes(self, mock_client):
        """is_controversial should return True when Gemini says YES."""
        from bias_agent import is_controversial
        mock_response = MagicMock()
        mock_response.text = "YES"
        mock_client.models.generate_content.return_value = mock_response
        assert is_controversial("gun control") is True

    @patch("bias_agent.client")
    def test_returns_false_for_no(self, mock_client):
        """is_controversial should return False when Gemini says NO."""
        from bias_agent import is_controversial
        mock_response = MagicMock()
        mock_response.text = "NO"
        mock_client.models.generate_content.return_value = mock_response
        assert is_controversial("how does photosynthesis work") is False

    @patch("bias_agent.client")
    def test_case_insensitive(self, mock_client):
        """is_controversial should handle lowercase yes."""
        from bias_agent import is_controversial
        mock_response = MagicMock()
        mock_response.text = "yes"
        mock_client.models.generate_content.return_value = mock_response
        assert is_controversial("abortion") is True


class TestAnalyzeBias:
    @patch("bias_agent.read_webpage")
    @patch("bias_agent.client")
    def test_returns_bias_dict(self, mock_client, mock_read):
        """analyze_bias should return a dict with required keys."""
        from bias_agent import analyze_bias
        mock_read.return_value = "Some article content here."
        mock_response = MagicMock()
        mock_response.text = "BIAS: LEFT\nREASON: Uses progressive framing."
        mock_client.models.generate_content.return_value = mock_response
        result = analyze_bias("Test Article", "https://example.com")
        assert "bias" in result
        assert "explanation" in result
        assert "title" in result
        assert "url" in result
        assert result["bias"] == "LEFT"

    @patch("bias_agent.read_webpage")
    @patch("bias_agent.client")
    def test_handles_unreadable_page(self, mock_client, mock_read):
        """analyze_bias should handle pages that can't be fetched."""
        from bias_agent import analyze_bias
        mock_read.return_value = "Could not read page: Timeout"
        result = analyze_bias("Bad Article", "https://broken.com")
        assert result["bias"] == "UNKNOWN"

    @patch("bias_agent.read_webpage")
    @patch("bias_agent.client")
    def test_passes_url_and_title(self, mock_client, mock_read):
        """analyze_bias result should contain the original title and url."""
        from bias_agent import analyze_bias
        mock_read.return_value = "Content"
        mock_response = MagicMock()
        mock_response.text = "BIAS: NEUTRAL\nREASON: Balanced reporting."
        mock_client.models.generate_content.return_value = mock_response
        result = analyze_bias("My Title", "https://myurl.com")
        assert result["title"] == "My Title"
        assert result["url"] == "https://myurl.com"


class TestRunBiasCheck:
    @patch("bias_agent.is_controversial")
    def test_returns_none_if_not_controversial(self, mock_controversial):
        """run_bias_check should return None for non-controversial topics."""
        from bias_agent import run_bias_check
        mock_controversial.return_value = False
        result = run_bias_check("how does gravity work", [])
        assert result is None

    @patch("bias_agent.analyze_bias")
    @patch("bias_agent.is_controversial")
    def test_analyzes_all_sources(self, mock_controversial, mock_analyze):
        """run_bias_check should call analyze_bias for each source."""
        from bias_agent import run_bias_check
        mock_controversial.return_value = True
        mock_analyze.return_value = {
            "title": "T", "url": "U", "bias": "NEUTRAL", "explanation": "E"
        }
        sources = [
            {"title": "A", "url": "https://a.com"},
            {"title": "B", "url": "https://b.com"},
        ]
        results = run_bias_check("gun control", sources)
        assert len(results) == 2
        assert mock_analyze.call_count == 2


# ─────────────────────────────────────────────
# main.py tests
# ─────────────────────────────────────────────

class TestBiasOptions:
    def test_all_five_options_exist(self):
        """All 5 bias options should be defined in main."""
        from main import BIAS_OPTIONS
        assert len(BIAS_OPTIONS) == 5
        assert "1" in BIAS_OPTIONS
        assert "5" in BIAS_OPTIONS

    def test_bias_colors_cover_all_options(self):
        """Every bias label should have a corresponding color/icon."""
        from main import BIAS_OPTIONS, BIAS_COLORS
        for label in BIAS_OPTIONS.values():
            assert label in BIAS_COLORS