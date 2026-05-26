from agent import research


def main():
    print("=" * 50)
    print("       Personal Research Agent (Gemini)")
    print("=" * 50)
    print("Ask any research question. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()

        if not question:
            continue

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        report = research(question)

        print("\n" + "=" * 50)
        print(report)
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()