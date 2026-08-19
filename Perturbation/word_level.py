import random
from typing import Dict, List

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from openai import OpenAI

for resource in [
    "stopwords", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "universal_tagset", "punkt", "punkt_tab",
]:
    nltk.download(resource, quiet=True)

STOPWORDS = {w.lower() for w in nltk_stopwords.words("english")}
STOPWORDS.update({
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
    "should", "will", "shall", "have", "has", "had", "may", "might",
})

LOW_CONTENT_POS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "DT", "PDT"}
FILLER_WORDS = {"just", "very", "really", "quite", "rather", "basically", "actually", "generally", "usually", "often", "simply", "a", "an", "the"}
NEUTRAL_WORDS = [
    "generally", "usually", "typically", "often", "commonly", "normally",
    "particularly", "especially", "primarily", "largely", "specific", "particular",
    "certain", "various", "general", "overall", "typical", "common", "relevant", "notable",
]

_DETOKENIZER = TreebankWordDetokenizer()


def set_seed(seed: int) -> None:
    random.seed(seed)


def _content_indices(words: List[str]) -> List[int]:
    return [i for i, word in enumerate(words) if word.isalpha() and word.lower() not in STOPWORDS]


def synonym_swap(question: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    client = OpenAI()
    prompt = (
        "Replace one content word in the following question with a natural, contextually appropriate synonym.\n"
        "Requirements:\n"
        "- Modify exactly one content word only.\n"
        "- Do not change stopwords, question words, or the sentence structure.\n"
        "- Return only the modified question with no explanation.\n"
        f"Question: {question}"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def neighbor_swap(question: str) -> str:
    words = question.split()
    content = _content_indices(words)
    adjacent_pairs = [
        (content[i], content[i + 1])
        for i in range(len(content) - 1)
        if content[i + 1] == content[i] + 1
    ]
    if not adjacent_pairs:
        return question
    i, j = random.choice(adjacent_pairs)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def neutral_insertion(question: str) -> str:
    words = question.split()
    content = _content_indices(words)
    if not content:
        return question
    insert_pos = random.choice(content)
    words.insert(insert_pos, random.choice(NEUTRAL_WORDS))
    return " ".join(words)


def neutral_deletion(question: str) -> str:
    tokens = nltk.word_tokenize(question)
    tagged = nltk.pos_tag(tokens)
    candidates = [
        i for i, (word, tag) in enumerate(tagged)
        if tag in LOW_CONTENT_POS or word.lower() in FILLER_WORDS
    ]
    if not candidates:
        candidates = _content_indices(tokens)
    if not candidates:
        return question
    delete_idx = random.choice(candidates)
    result = [word for i, word in enumerate(tokens) if i != delete_idx]
    return _DETOKENIZER.detokenize(result)


def perturb_question(question: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> Dict[str, str]:
    return {
        "original": question,
        "SynonymSwap": synonym_swap(question, model, temperature),
        "NeighborSwap": neighbor_swap(question),
        "NeutralInsertion": neutral_insertion(question),
        "NeutralDeletion": neutral_deletion(question),
    }
