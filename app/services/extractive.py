import math
import re
from collections import Counter

from app.config import settings
from app.services.base import BaseSummarizer

_CONTROL_CHAR_PATTERN = re.compile(r"[^\x20-\x7E\s]")


class ExtractiveSummarizer(BaseSummarizer):
    """Extractive summarizer using TF-IDF sentence scoring.

    Selects the most important sentences from the original text based on
    term frequency-inverse document frequency scores. No ML model required.
    """

    STOP_WORDS: frozenset[str] = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "because",
            "but",
            "and",
            "or",
            "if",
            "while",
            "although",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "he",
            "she",
            "they",
            "them",
            "his",
            "her",
            "their",
            "we",
            "our",
            "you",
            "your",
            "what",
            "which",
            "who",
            "whom",
        }
    )

    _SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
    _WORD_PATTERN = re.compile(r"\b[a-zA-Z]{2,}\b")

    @staticmethod
    def _sanitize(text: str) -> str:
        """Filter control characters, keeping only printable + whitespace."""
        return _CONTROL_CHAR_PATTERN.sub("", text)

    def summarize(self, text: str, **kwargs) -> str:
        text = self._sanitize(text)
        num_sentences = kwargs.get("num_sentences") or settings.extractive_sentence_count
        sentences = self._split_sentences(text)

        if len(sentences) <= num_sentences:
            return text.strip()

        scores = self._score_sentences(sentences)
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        selected = sorted(ranked[:num_sentences])

        return " ".join(sentences[i] for i in selected)

    def _split_sentences(self, text: str) -> list[str]:
        sentences = self._SENTENCE_PATTERN.split(text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _tokenize(self, text: str) -> list[str]:
        words = self._WORD_PATTERN.findall(text.lower())
        return [w for w in words if w not in self.STOP_WORDS]

    def _score_sentences(self, sentences: list[str]) -> list[float]:
        doc_tokens = [self._tokenize(s) for s in sentences]
        num_docs = len(sentences)

        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            for word in set(tokens):
                df[word] += 1

        scores = []
        for tokens in doc_tokens:
            if not tokens:
                scores.append(0.0)
                continue
            tf = Counter(tokens)
            score = sum((tf[w] / len(tokens)) * math.log((num_docs + 1) / (df[w] + 1)) for w in tf)
            scores.append(score)

        return scores
