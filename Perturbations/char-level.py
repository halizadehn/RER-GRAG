import nltk
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("universal_tagset", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords as nltk_stopwords

from textattack.augmentation import Augmenter
from textattack.transformations import (
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterSubstitution,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,
)

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

        # Extend with question words and auxiliaries not always in NLTK list
        self.stopwords.update({
            "what", "which", "who", "whom", "whose",
            "when", "where", "why", "how",
            "is", "are", "was", "were",
            "do", "does", "did",
            "can", "could", "would", "should", "will", "shall",
            "have", "has", "had", "may", "might",
        })

    def _get_modifiable_indices(self, current_text):
        """Return indices of tokens that are NOT stopwords."""
        return {
            i
            for i, word in enumerate(current_text.words)
            if word.lower() not in self.stopwords
        }

STOPWORD_CONSTRAINT = CaseInsensitiveStopwordConstraint()

def rand_delete(question: str, pct_words_to_swap: float = 0.2) -> str:

    augmenter = Augmenter(
        transformation=WordSwapRandomCharacterDeletion(),
        constraints=[STOPWORD_CONSTRAINT],
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=1,
    )
    results = augmenter.augment(question)
    return results[0] if results else question


def rand_sub(question: str, pct_words_to_swap: float = 0.2) -> str:

    augmenter = Augmenter(
        transformation=WordSwapRandomCharacterSubstitution(),
        constraints=[STOPWORD_CONSTRAINT],
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=1,
    )
    results = augmenter.augment(question)
    return results[0] if results else question


def swap_neighbor(question: str, pct_words_to_swap: float = 0.2) -> str:

    augmenter = Augmenter(
        transformation=WordSwapNeighboringCharacterSwap(),
        constraints=[STOPWORD_CONSTRAINT],
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=1,
    )
    results = augmenter.augment(question)
    return results[0] if results else question


def swap_adjacent_keyboard(question: str, pct_words_to_swap: float = 0.2) -> str:

    augmenter = Augmenter(
        transformation=WordSwapQWERTY(),
        constraints=[STOPWORD_CONSTRAINT],
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=1,
    )
    results = augmenter.augment(question)
    return results[0] if results else question


def perturb_question(question: str, pct_words_to_swap: float = 0.2) -> dict:
    return {
        "original":      question,
        "RandDelete":    rand_delete(question, pct_words_to_swap),
        "RandSub":       rand_sub(question, pct_words_to_swap),
        "SwapNeighbor":  swap_neighbor(question, pct_words_to_swap),
        "SwapAdjacent":  swap_adjacent_keyboard(question, pct_words_to_swap),
    }


def display_results(perturbations: dict) -> None:
    width = 65
    print("\n" + "=" * width)
    print(" QUERY PERTURBATION RESULTS ".center(width))
    print(" (stopwords protected — case-insensitive) ".center(width))
    print("=" * width)

    labels = {
        "original":     "Original",
        "RandDelete":   "RandDelete   (random char deletion)",
        "RandSub":      "RandSub      (random char substitution)",
        "SwapNeighbor": "SwapNeighbor (swap neighboring chars)",
        "SwapAdjacent": "SwapAdjacent (swap keyboard-adjacent char)",
    }

    for key, label in labels.items():
        prefix = ">" if key != "original" else " "
        print(f"\n  {prefix} {label}")
        print(f"    {perturbations[key]}")

    print("\n" + "=" * width + "\n")
