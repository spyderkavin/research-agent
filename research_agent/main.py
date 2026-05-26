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


def print_bias_results(bias_results: list[dict]):
    print("\n" + "=" * 50)
    print("       BIAS ANALYSIS")
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

        # Step 1 — Run research agent
        report, sources = research(question)

        # Step 2 — Print the report
        print("\n" + "=" * 50)
        print(report)
        print("=" * 50)

        # Step 3 — Run bias check if needed
        if sources:
            bias_results = run_bias_check(question, sources)
            if bias_results:
                print_bias_results(bias_results)
        else:
            print("\n⚠️  No sources found to analyze.\n")


if __name__ == "__main__":
    main()