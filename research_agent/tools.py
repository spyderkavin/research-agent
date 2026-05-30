import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import json


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information on a topic.

    Args:
        query: The search query to run.
        max_results: Number of results to return, default is 5.

    Returns:
        A JSON string containing search results with titles, URLs, and snippets.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        formatted = [
            {
                "title": r["title"],
                "url": r["href"],
                "snippet": r["body"]
            }
            for r in results
        ]
        return json.dumps(formatted, indent=2)
    except Exception as e:
        return f"Search failed: {str(e)}"


def read_webpage(url: str) -> str:
    """Read the full text content of a webpage by its URL.

    Args:
        url: The full URL of the webpage to read.

    Returns:
        The cleaned text content of the webpage.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text[:4000] if len(text) > 4000 else text

    except Exception as e:
        return f"Could not read page: {str(e)}"