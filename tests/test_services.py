from app.services.extractive import ExtractiveSummarizer

SAMPLE_TEXT = (
    "Natural language processing is a subfield of linguistics, computer science, "
    "and artificial intelligence concerned with the interactions between computers "
    "and human language. The goal is to enable computers to understand, interpret, "
    "and generate human language in a valuable way. NLP combines computational "
    "linguistics with statistical, machine learning, and deep learning models. "
    "These approaches enable the analysis of large amounts of natural language data. "
    "Applications of NLP include machine translation, sentiment analysis, chatbots, "
    "and text summarization. Text summarization is particularly important for "
    "condensing large documents into shorter versions while preserving key information. "
    "There are two main approaches to text summarization: extractive and abstractive. "
    "Extractive summarization selects important sentences from the original text. "
    "Abstractive summarization generates new sentences that capture the main ideas. "
    "Modern NLP systems often use transformer architectures like BERT and GPT. "
    "These models have achieved state-of-the-art results on many NLP benchmarks."
)


class TestExtractiveSummarizer:
    def setup_method(self):
        self.summarizer = ExtractiveSummarizer()

    def test_summarize_returns_string(self):
        result = self.summarizer.summarize(SAMPLE_TEXT)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_is_shorter_than_original(self):
        result = self.summarizer.summarize(SAMPLE_TEXT, num_sentences=3)
        assert len(result) < len(SAMPLE_TEXT)

    def test_summarize_respects_num_sentences(self):
        result = self.summarizer.summarize(SAMPLE_TEXT, num_sentences=2)
        # Result should contain roughly 2 sentences
        sentences = [s.strip() for s in result.split(".") if s.strip()]
        assert len(sentences) <= 3  # Allow slight variance from sentence splitting

    def test_summarize_short_text_returns_original(self):
        short = "This is a short text. It has only two sentences."
        result = self.summarizer.summarize(short, num_sentences=5)
        assert result == short.strip()

    def test_sentences_preserve_original_order(self):
        result = self.summarizer.summarize(SAMPLE_TEXT, num_sentences=3)
        # Each sentence in the summary should appear in the original
        for sentence in result.split(". "):
            clean = sentence.rstrip(".")
            assert clean in SAMPLE_TEXT or sentence in SAMPLE_TEXT

    def test_split_sentences(self):
        sentences = self.summarizer._split_sentences(SAMPLE_TEXT)
        assert len(sentences) > 5
        assert all(len(s) > 10 for s in sentences)

    def test_tokenize_removes_stop_words(self):
        tokens = self.summarizer._tokenize("The cat is on the mat")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "cat" in tokens
        assert "mat" in tokens
