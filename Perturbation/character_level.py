import random
from typing import Dict

import nltk
import numpy as np

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
        self.stopwords = {w.lower() for w in nltk_stopwords.words(language)}
        self.stopwords.update({
            "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
            "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
            "should", "will", "shall", "have", "has", "had", "may", "might",
        })

    def _get_modifiable_indices(self, current_text):
        return {i for i, word in enumerate(current_text.words) if word.lower() not in self.stopwords}


STOPWORD_CONSTRAINT = CaseInsensitiveStopwordConstraint()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _augment(question: str, transformation, pct_words_to_swap: float = 0.2) -> str:
    augmenter = Augmenter(
        transformation=transformation,
        constraints=[STOPWORD_CONSTRAINT],
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=1,
    )
    results = augmenter.augment(question)
    return results[0] if results else question


def rand_swap(question: str, pct_words_to_swap: float = 0.2) -> str:
    return _augment(question, WordSwapRandomCharacterSubstitution(), pct_words_to_swap)


def neighbor_swap(question: str, pct_words_to_swap: float = 0.2) -> str:
    return _augment(question, WordSwapNeighboringCharacterSwap(), pct_words_to_swap)


def adjacent_swap(question: str, pct_words_to_swap: float = 0.2) -> str:
    return _augment(question, WordSwapQWERTY(), pct_words_to_swap)


def rand_deletion(question: str, pct_words_to_swap: float = 0.2) -> str:
    return _augment(question, WordSwapRandomCharacterDeletion(), pct_words_to_swap)


def perturb_question(question: str, pct_words_to_swap: float = 0.2) -> Dict[str, str]:
    return {
        "original": question,
        "RandSwap": rand_swap(question, pct_words_to_swap),
        "NeighborSwap": neighbor_swap(question, pct_words_to_swap),
        "AdjacentSwap": adjacent_swap(question, pct_words_to_swap),
        "RandDeletion": rand_deletion(question, pct_words_to_swap),
    }
