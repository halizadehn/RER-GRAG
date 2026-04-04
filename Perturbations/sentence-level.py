from openai import OpenAI

client = OpenAI()

def _call(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def paraphrase(question: str) -> str:
    return _call(
        f"Rewrite the following question by expressing the same information need using different vocabulary, "
        f"sentence structure, and phrasing. The rewritten question must preserve the original intent and factual "
        f"scope exactly — do not add, remove, or alter any information being asked. Avoid simply substituting "
        f"individual words; aim for a structurally distinct reformulation that a different user might naturally write. "
        f"Return only the rewritten question with no explanation.\nQuestion: {question}"
    )


def formality(question: str) -> str:
    return _call(
        f"Rewrite the following question in an informal, colloquial style as it would appear in a casual online "
        f"conversation or chat message. Use contractions, informal vocabulary, and conversational expressions "
        f"(e.g., 'what's the deal with', 'like', 'kinda', 'gonna'). Do not change the information being requested "
        f"— only the register and tone should change. Return only the rewritten question with no explanation.\n"
        f"Question: {question}"
    )


def readability(question: str) -> str:
    return _call(
        f"Rewrite the following question to make it less readable by increasing its linguistic complexity. Use longer, "
        f"more convoluted sentence structures, nominalization, passive constructions, and low-frequency vocabulary "
        f"where appropriate. The rewritten question must ask for exactly the same information as the original. "
        f"Do not change the topic or scope. Return only the rewritten question with no explanation.\n"
        f"Question: {question}"
    )


def politeness(question: str) -> str:
    return _call(
        "Task: Rewrite Queries to Sound More Polite and Courteous.\n"
        "Rephrase the given query into a more polite, respectful, and considerate version while preserving its original meaning and intent.\n"
        "The output should sound well-mannered for professional or friendly interaction and MUST be a single sentence.\n"
        "Critical Rules:\n"
        "- Maintain the original meaning and intent without unnecessary embellishment.\n"
        "- Use politeness naturally and vary it across queries (randomly choose different wording/techniques).\n"
        "- Avoid robotic or overly formal constructions.\n"
        "- Do NOT start with 'I hope you are ...'.\n"
        "- Do NOT end with a standalone 'Thank you' sentence.\n"
        "Generate ONLY the single polite query sentence.\n\n"
        f"Question: {question}"
    )


def perturb_question_sentence_level(question: str) -> dict:
    return {
        "original":    question,
        "Paraphrase":  paraphrase(question),
        "Formality":   formality(question),
        "Readability": readability(question),
        "Politeness":  politeness(question),
    }


def display_results(perturbations: dict) -> None:
    width = 70
    print("\n" + "=" * width)
    print(" SENTENCE-LEVEL PERTURBATION RESULTS ".center(width))
    print("=" * width)

    labels = {
        "original":    ("Original",    ""),
        "Paraphrase":  ("Paraphrase",  "(structural reformulation — same intent)"),
        "Formality":   ("Formality",   "(informal/colloquial register — Cao et al. 2025)"),
        "Readability": ("Readability", "(increased linguistic complexity — Cao et al. 2025)"),
        "Politeness":  ("Politeness",  "(courteous framing — Cao et al. 2025)"),
    }

    for key, (name, note) in labels.items():
        prefix = ">" if key != "original" else " "
        header = f"{name}  {note}".strip()
        print(f"\n  {prefix} {header}")
        print(f"    {perturbations[key]}")

    print("\n" + "=" * width + "\n")
