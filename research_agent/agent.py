from google import genai
from google.genai import types
import os, json, re
from dotenv import load_dotenv
from tools import web_search, read_webpage

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM_PROMPT = """You are a thorough research agent. When given a research question:

1. Break it into 2-3 focused sub-questions
2. Search the web for each sub-question
3. Read the most relevant pages in full for detail
4. Synthesize everything into a structured final report

Format your final report exactly like this:

## Summary
(2-3 sentence overview)

## Key Findings
- Finding 1
- Finding 2
- ...

## Sources
- [Title](URL)
- [Title](URL)
- ...

Only state things you found in your research. Always include every source you used.
"""

TOOLS = [web_search, read_webpage]


def run_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return web_search(args["query"], args.get("max_results", 5))
    elif name == "read_webpage":
        return read_webpage(args["url"])
    return f"Unknown tool: {name}"


def extract_sources(report: str) -> list[dict]:
    """Pull out sources from the report's ## Sources section."""
    sources = []
    in_sources = False
    for line in report.split("\n"):
        if "## Sources" in line:
            in_sources = True
            continue
        if in_sources and line.startswith("- "):
            # Parse markdown links: [Title](URL)
            match = re.search(r'\[(.+?)\]\((https?://[^\)]+)\)', line)
            if match:
                sources.append({
                    "title": match.group(1),
                    "url": match.group(2)
                })
    return sources


def research(question: str) -> tuple[str, list[dict]]:
    """
    Run the research agent.
    Returns (report_text, list of sources)
    """
    print(f"\n🔍 Researching: {question}\n")

    messages = [types.Content(
        role="user",
        parts=[types.Part(text=question)]
    )]

    while True:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
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