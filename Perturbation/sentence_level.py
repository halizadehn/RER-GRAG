from typing import Dict
from openai import OpenAI


def _call(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def paraphrase(question: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    prompt = (
        "Rewrite the question below while preserving its exact meaning.\n"
        "Requirements:\n"
        "- Use different vocabulary, phrasing, and sentence structure.\n"
        "- Preserve the original intent and factual scope exactly.\n"
        "- Do not add, remove, or alter any information.\n"
        "- Avoid simple word-by-word substitution.\n"
        "- Produce a structurally distinct reformulation that a different user might naturally write.\n"
        "- Return only the rewritten question with no explanation.\n"
        f"Question: {question}"
    )
    return _call(prompt, model, temperature)


def formality(question: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    prompt = (
        "Rewrite the question below in an informal, conversational style, as if written in a casual online chat.\n"
        "Requirements:\n"
        "- Use contractions, informal vocabulary, and conversational expressions.\n"
        "- Keep the exact same information request.\n"
        "- Do not add, remove, or change any meaning—only tone/register.\n"
        "- Return only the rewritten question with no explanation.\n"
        f"Question: {question}"
    )
    return _call(prompt, model, temperature)


def politeness(question: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    prompt = (
        "Rewrite the following question to sound more polite and courteous.\n"
        "Requirements:\n"
        "- Rephrase the given query into a more polite, respectful, and considerate version while preserving its original meaning and intent.\n"
        "- Maintain the original meaning and intent without unnecessary embellishment.\n"
        "- Use politeness naturally and randomly choose different wording/techniques.\n"
        "- The output should sound well-mannered for professional or friendly interaction.\n"
        "- Return only the rewritten question with no explanation.\n"
        f"Question: {question}"
    )
    return _call(prompt, model, temperature)


def readability(question: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    prompt = (
        "Rewrite the following question to make it less readable by increasing its linguistic complexity.\n"
        "Requirements:\n"
        "- Use longer, more convoluted sentence structures, nominalization, passive constructions, and low-frequency vocabulary where appropriate.\n"
        "- The rewritten question must ask for exactly the same information as the original.\n"
        "- Do not change the topic or scope.\n"
        "- Return only the rewritten question with no explanation.\n"
        f"Question: {question}"
    )
    return _call(prompt, model, temperature)


def perturb_question(question: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> Dict[str, str]:
    return {
        "original": question,
        "Paraphrase": paraphrase(question, model, temperature),
        "Formality": formality(question, model, temperature),
        "Politeness": politeness(question, model, temperature),
        "Readability": readability(question, model, temperature),
    }
