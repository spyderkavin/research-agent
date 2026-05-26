from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from tools import read_webpage

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def is_controversial(topic: str) -> bool:
    """
    Ask Gemini if a topic is even 1% controversial.
    Returns True if it is, False if completely neutral.
    """
    prompt = f"""Is the following topic controversial in any way — politically, 
socially, scientifically, ethically, or culturally? 
Even if it is only slightly controversial (1% or more), answer YES.
Only answer NO if the topic is completely factual and neutral with zero controversy.

Topic: {topic}

Reply with only YES or NO. Nothing else."""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    answer = response.text.strip().upper()
    return "YES" in answer


def analyze_bias(title: str, url: str) -> dict:
    """
    Read a source and analyze it for political/ideological bias.
    Returns a dict with bias direction and explanation.
    """
    print(f"  🔎 Analyzing bias: {title[:60]}...")

    # Read the article content
    content = read_webpage(url)

    if "Could not read page" in content:
        return {
            "title": title,
            "url": url,
            "bias": "UNKNOWN",
            "explanation": "Could not access article content."
        }

    prompt = f"""Analyze the following article for bias. Determine if it leans:
- LEFT (liberal, progressive bias)
- CENTER-LEFT (slight liberal lean)  
- NEUTRAL (balanced, factual, no clear lean)
- CENTER-RIGHT (slight conservative lean)
- RIGHT (conservative bias)

Look for: loaded language, selective facts, framing, omitted perspectives,
emotional appeals, and which side gets more favorable coverage.

Article title: {title}
Article content:
{content[:3000]}

Respond in exactly this format:
BIAS: [LEFT / CENTER-LEFT / NEUTRAL / CENTER-RIGHT / RIGHT]
REASON: [One sentence explaining why]"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )

    result = response.text.strip()

    # Parse the response
    bias = "UNKNOWN"
    reason = "Could not determine."

    for line in result.split("\n"):
        if line.startswith("BIAS:"):
            bias = line.replace("BIAS:", "").strip()
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return {
        "title": title,
        "url": url,
        "bias": bias,
        "explanation": reason
    }


def run_bias_check(topic: str, sources: list[dict]) -> list[dict] | None:
    """
    Main entry point. Checks if topic is controversial,
    then analyzes each source if it is.
    Returns list of bias results, or None if not controversial.
    """
    print(f"\n🧭 Checking if topic is controversial...")

    if not is_controversial(topic):
        print("  ✅ Topic is not controversial — skipping bias analysis.\n")
        return None

    print(f"  ⚠️  Topic is controversial — running bias analysis on {len(sources)} sources...\n")

    results = []
    for source in sources:
        result = analyze_bias(source["title"], source["url"])
        results.append(result)

    return results