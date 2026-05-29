from google import genai
from google.genai import types
import os, re
from dotenv import load_dotenv
from pathlib import Path
from tools import web_search, read_webpage

load_dotenv(Path(__file__).parent.parent / ".env")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

TOOLS = [web_search, read_webpage]

BIAS_INSTRUCTIONS = {
    "LEFT": """
You are researching from a PROGRESSIVE/LEFT-LEANING perspective.
- Prioritize sources from progressive outlets
- Frame findings around social equity, systemic issues, and collective solutions
- Highlight impacts on marginalized communities
- Emphasize government and institutional responsibility
""",
    "CENTER-LEFT": """
You are researching from a CENTER-LEFT perspective.
- Use mostly mainstream and slightly progressive sources
- Present balanced findings but lean toward progressive conclusions where evidence supports it
- Acknowledge multiple perspectives but give slightly more weight to liberal viewpoints
""",
    "NEUTRAL": """
You are researching from a NEUTRAL, UNBIASED perspective.
- Use balanced sources from across the political spectrum
- Present all sides equally with no favored framing
- Stick strictly to facts and let the reader draw their own conclusions
- Actively avoid loaded language
""",
    "CENTER-RIGHT": """
You are researching from a CENTER-RIGHT perspective.
- Use mostly mainstream and slightly conservative sources
- Present balanced findings but lean toward conservative conclusions where evidence supports it
- Acknowledge multiple perspectives but give slightly more weight to conservative viewpoints
""",
    "RIGHT": """
You are researching from a CONSERVATIVE/RIGHT-LEANING perspective.
- Prioritize sources from conservative outlets
- Frame findings around personal responsibility, free markets, and traditional values
- Emphasize individual impact over systemic explanations
- Highlight concerns about government overreach where relevant
"""
}

BASE_SYSTEM_PROMPT = """You are a thorough research agent. When given a research question:

1. Break it into 2-3 focused sub-questions
2. Search the web for each sub-question, choosing sources that match your perspective
3. Read the most relevant pages in full for detail
4. Synthesize everything into a structured final report that reflects your assigned perspective

Format your final report exactly like this:

## Summary
(2-3 sentence overview written from your assigned perspective)

## Key Findings
- Finding 1
- Finding 2
- ...

## Sources
- [Title](URL)
- [Title](URL)
- ...

Always include every source you used. Only state things you found in your research.
"""


def build_system_prompt(bias_preference: str) -> str:
    bias_instruction = BIAS_INSTRUCTIONS.get(bias_preference, BIAS_INSTRUCTIONS["NEUTRAL"])
    return BASE_SYSTEM_PROMPT + "\n## Your Perspective\n" + bias_instruction


def extract_sources(report: str) -> list[dict]:
    sources = []
    in_sources = False
    for line in report.split("\n"):
        if "## Sources" in line:
            in_sources = True
            continue
        if in_sources and line.startswith("- "):
            match = re.search(r'\[(.+?)\]\((https?://[^\)]+)\)', line)
            if match:
                sources.append({
                    "title": match.group(1),
                    "url": match.group(2)
                })
    return sources


def run_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return web_search(args["query"], args.get("max_results", 5))
    elif name == "read_webpage":
        return read_webpage(args["url"])
    return f"Unknown tool: {name}"


def research(question: str, bias_preference: str = "NEUTRAL") -> tuple[str, list[dict]]:
    print(f"\n🔍 Researching: {question}")
    print(f"🧭 Perspective: {bias_preference}\n")

    system_prompt = build_system_prompt(bias_preference)

    messages = [types.Content(
        role="user",
        parts=[types.Part(text=question)]
    )]

    while True:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=TOOLS,
            )
        )

        messages.append(types.Content(
            role="model",
            parts=response.candidates[0].content.parts
        ))

        fn_calls = [
            part for part in response.candidates[0].content.parts
            if part.function_call is not None
        ]

        if not fn_calls:
            report = response.text
            sources = extract_sources(report)
            return report, sources

        tool_results = []
        for part in fn_calls:
            name = part.function_call.name
            args = dict(part.function_call.args)
            print(f"  🛠  {name}({str(args)[:80]})")
            result = run_tool(name, args)
            tool_results.append(types.Part(
                function_response=types.FunctionResponse(
                    name=name,
                    response={"result": result}
                )
            ))

        messages.append(types.Content(
            role="user",
            parts=tool_results
        ))