import nltk
nltk.download("stopwords",                      quiet=True)
nltk.download("wordnet",                        quiet=True)
nltk.download("omw-1.4",                        quiet=True)
nltk.download("averaged_perceptron_tagger",     quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("universal_tagset",               quiet=True)
nltk.download("punkt",                          quiet=True)
nltk.download("punkt_tab",                      quiet=True)
from openai import OpenAI
client = OpenAI()
import random
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet
import nltk
from textattack.augmentation import Augmenter
from textattack.transformations import WordSwapWordNet
try:
    from textattack.constraints.pre_transformation import PreTransformationConstraint as _Base
except ImportError:
    try:
        from textattack.constraints import PreTransformationConstraint as _Base
    except ImportError:
        class _Base:
            def _get_modifiable_indices(self, current_text):
                raise NotImplementedError
            def __call__(self, current_text, *args, **kwargs):
                return self._get_modifiable_indices(current_text)

class CaseInsensitiveStopwordConstraint(_Base):

    def __init__(self, language: str = "english"):
        self.stopwords = set(w.lower() for w in nltk_stopwords.words(language))
        self.stopwords.update({
            "what", "which", "who", "whom", "whose",
            "when", "where", "why", "how",
            "is", "are", "was", "were",
            "do", "does", "did",
            "can", "could", "would", "should", "will", "shall",
            "have", "has", "had", "may", "might",
        })

    def _get_modifiable_indices(self, current_text):
        return {
            i for i, word in enumerate(current_text.words)
            if word.lower() not in self.stopwords
        }

STOPWORD_CONSTRAINT = CaseInsensitiveStopwordConstraint()
STOPWORDS = STOPWORD_CONSTRAINT.stopwords

def _content_indices(words: list[str]) -> list[int]:
    return [i for i, w in enumerate(words) if w.lower() not in STOPWORDS]

def _get_wordnet_synonym(word: str) -> str | None:
    candidates = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower():
                candidates.append(candidate)
    return random.choice(candidates) if candidates else None

def synonym_replacement_v0(question: str) -> str:
    words = question.split()
    candidates = _content_indices(words)
    random.shuffle(candidates)

    for i in candidates:
        syn = _get_wordnet_synonym(words[i])
        if syn:
            words[i] = syn
            return " ".join(words)

    return question

def synonym_replacement(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[{"role": "user", "content":
            f"Replace exactly one content word in the following question with a natural, "
            f"contextually appropriate synonym. Do not change stopwords, question words, or "
            f"the sentence structure. Return only the modified question with no explanation.\n"
            f"Question: {question}"
        }],
    )
    return response.choices[0].message.content.strip()


def word_order(question: str) -> str:
    words = question.split()
    content = _content_indices(words)
    adjacent_pairs = [
        (content[i], content[i + 1])
        for i in range(len(content) - 1)
        if content[i + 1] == content[i] + 1
    ]

    if not adjacent_pairs:
        if len(content) >= 2:
            i, j = random.sample(content, 2)
            words[i], words[j] = words[j], words[i]
            return " ".join(words)
        return question

    i, j = random.choice(adjacent_pairs)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)

VERB_SINGULAR_TO_PLURAL = {
    "is": "are", "was": "were", "has": "have",
    "does": "do", "goes": "go", "says": "say",
    "makes": "make", "takes": "take", "causes": "cause",
    "affects": "affect", "changes": "change", "needs": "need",
    "includes": "include", "requires": "require", "helps": "help",
    "seems": "seem", "becomes": "become", "shows": "show",
    "means": "mean", "leads": "lead", "allows": "allow",
    "refers": "refer", "works": "work", "differs": "differ",
    "defines": "define", "describes": "describe", "explains": "explain",
    "represents": "represent", "contains": "contain", "provides": "provide",
    "suggests": "suggest", "indicates": "indicate", "supports": "support",
}
VERB_PLURAL_TO_SINGULAR = {v: k for k, v in VERB_SINGULAR_TO_PLURAL.items()}

def _flip_verb(word: str) -> str | None:
    lw = word.lower()
    if lw in VERB_SINGULAR_TO_PLURAL:
        flipped = VERB_SINGULAR_TO_PLURAL[lw]
    elif lw in VERB_PLURAL_TO_SINGULAR:
        flipped = VERB_PLURAL_TO_SINGULAR[lw]
    elif lw.endswith("s") and not lw.endswith(("ss", "us", "is", "as", "ous")):
        flipped = lw[:-1]
    elif not lw.endswith("s"):
        flipped = lw + "s"
    else:
        return None
    return flipped.capitalize() if word[0].isupper() else flipped

def singular_plural_verb(question: str) -> str:
    tokens = nltk.word_tokenize(question)
    tagged = nltk.pos_tag(tokens)
    verb_candidates = [
        i for i, (word, tag) in enumerate(tagged)
        if tag in ("VBZ", "VBP")
    ]

    random.shuffle(verb_candidates)
    for i in verb_candidates:
        flipped = _flip_verb(tokens[i])
        if flipped:
            tokens[i] = flipped
            return " ".join(tokens)

    return question

LOW_CONTENT_POS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "DT", "PDT"}
FILLER_WORDS = {
    "just", "very", "really", "quite", "rather", "basically",
    "actually", "generally", "usually", "often", "simply",
    "a", "an", "the",
}

def word_delete(question: str) -> str:
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

    to_delete = random.choice(candidates)
    result = [w for i, w in enumerate(tokens) if i != to_delete]
    return " ".join(result)

NEUTRAL_WORDS = [
    "generally", "usually", "typically", "often", "commonly",
    "normally", "particularly", "especially", "primarily", "largely",
    "specific", "particular", "certain", "various", "general",
    "overall", "typical", "common", "relevant", "notable",
]

def word_add(question: str) -> str:
    words = question.split()
    content = _content_indices(words)

    if not content:
        return question

    insert_pos = random.choice(content)
    neutral = random.choice(NEUTRAL_WORDS)
    words.insert(insert_pos, neutral)
    return " ".join(words)

def perturb_question_word_level(question: str) -> dict:
    return {
        "original":           question,
        "SynonymReplacement": synonym_replacement(question),
        "WordOrder":          word_order(question),
        "SingularPluralVerb": singular_plural_verb(question),
        "WordDelete":         word_delete(question),
        "WordAdd":            word_add(question),
    }

def display_results(perturbations: dict) -> None:
    width = 70
    print("\n" + "=" * width)
    print(" WORD-LEVEL PERTURBATION RESULTS ".center(width))
    print(" (exactly 1 natural change per strategy) ".center(width))
    print("=" * width)

    labels = {
        "original":           ("Original",           ""),
        "SynonymReplacement": ("SynonymReplacement", "(1 word -> WordNet synonym — EMNLP 2021)"),
        "WordOrder":          ("WordOrder",           "(swap 2 adjacent words — EMNLP 2021)"),
        "SingularPluralVerb": ("SingularPluralVerb",  "(flip 1 verb number — EMNLP 2021)"),
        "WordDelete":         ("WordDelete",          "(delete 1 low-content word — OpenReview)"),
        "WordAdd":            ("WordAdd",             "(insert 1 neutral word — OpenReview)"),
    }

    for key, (name, note) in labels.items():
        prefix = ">" if key != "original" else " "
        header = f"{name}  {note}".strip()
        print(f"\n  {prefix} {header}")
        print(f"    {perturbations[key]}")

    print("\n" + "=" * width + "\n")