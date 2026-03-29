import math
import re
from collections import Counter

from app.config import settings
from app.services.base import BaseSummarizer


class ExtractiveSummarizer(BaseSummarizer):
    """Extractive summarizer using TF-IDF sentence scoring.

    Selects the most important sentences from the original text based on
    term frequency-inverse document frequency scores. No ML model required.
    """

    def summarize(self, text: str, **kwargs) -> str:
        num_sentences = kwargs.get("num_sentences") or settings.extractive_sentence_count
        sentences = self._split_sentences(text)

        if len(sentences) <= num_sentences:
            return text.strip()

        scores = self._score_sentences(sentences)
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        selected = sorted(ranked[:num_sentences])

        return " ".join(sentences[i] for i in selected)

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "because", "but", "and", "or", "if", "while", "although", "this",
            "that", "these", "those", "it", "its", "he", "she", "they", "them",
            "his", "her", "their", "we", "our", "you", "your", "what", "which",
            "who", "whom",
        }
        return [w for w in words if w not in stop_words]

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
            score = sum(
                (tf[w] / len(tokens)) * math.log((num_docs + 1) / (df[w] + 1))
                for w in tf
            )
            scores.append(score)

        return scores
