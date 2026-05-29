from agent import research
from bias_agent import run_bias_check

BIAS_COLORS = {
    "LEFT": "🔵",
    "CENTER-LEFT": "🔷",
    "NEUTRAL": "⚪",
    "CENTER-RIGHT": "🔶",
    "RIGHT": "🔴",
    "UNKNOWN": "❓"
}

BIAS_OPTIONS = {
    "1": "LEFT",
    "2": "CENTER-LEFT",
    "3": "NEUTRAL",
    "4": "CENTER-RIGHT",
    "5": "RIGHT"
}

BIAS_DESCRIPTIONS = {
    "LEFT":         "Progressive/liberal framing, emphasizes social equity and systemic issues",
    "CENTER-LEFT":  "Slightly liberal lean, mostly balanced but favors progressive conclusions",
    "NEUTRAL":      "Purely factual, balanced perspectives from all sides",
    "CENTER-RIGHT": "Slightly conservative lean, mostly balanced but favors traditional conclusions",
    "RIGHT":        "Conservative framing, emphasizes personal responsibility and traditional values"
}


def ask_bias_preference() -> str:
    print("\n" + "=" * 50)
    print("  How would you like the research framed?")
    print("=" * 50)
    for key, label in BIAS_OPTIONS.items():
        desc = BIAS_DESCRIPTIONS[label]
        print(f"  {key}. {label:<15} — {desc}")
    print()

    while True:
        choice = input("Enter 1-5: ").strip()
        if choice in BIAS_OPTIONS:
            selected = BIAS_OPTIONS[choice]
            print(f"\n✅ Got it — researching with a {selected} perspective.\n")
            return selected
        print("Please enter a number between 1 and 5.")


def print_bias_results(bias_results: list[dict]):
    print("\n" + "=" * 50)
    print("       BIAS ANALYSIS OF SOURCES")
    print("=" * 50)
    for result in bias_results:
        icon = BIAS_COLORS.get(result["bias"], "❓")
        print(f"\n{icon}  {result['bias']}")
        print(f"   📰 {result['title'][:70]}")
        print(f"   💬 {result['explanation']}")
        print(f"   🔗 {result['url'][:70]}")
    print()


def main():
    print("=" * 50)
    print("   Personal Research Agent + Bias Detector")
    print("=" * 50)
    print("Ask any research question. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        bias_preference = ask_bias_preference()
        report, sources = research(question, bias_preference)

        print("\n" + "=" * 50)
        print(report)
        print("=" * 50)

        if sources:
            bias_results = run_bias_check(question, sources)
            if bias_results:
                print_bias_results(bias_results)
        else:
            print("\n⚠️  No sources found to analyze.\n")


if __name__ == "__main__":
    main()