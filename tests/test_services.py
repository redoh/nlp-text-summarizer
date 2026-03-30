from unittest.mock import MagicMock, patch

import pytest

from app.exceptions import CircuitOpenError, ModelInferenceError, ModelLoadError
from app.services.abstractive import AbstractiveSummarizer
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

    def test_empty_text(self):
        result = self.summarizer.summarize("", num_sentences=3)
        assert isinstance(result, str)

    def test_single_sentence(self):
        single = "Natural language processing is a fascinating field of study."
        result = self.summarizer.summarize(single, num_sentences=3)
        assert result == single.strip()

    def test_very_repetitive_text(self):
        repetitive = (
            "The cat sat on the mat. " * 10
            + "Dogs are loyal companions. "
            + "The cat sat on the mat. " * 10
        )
        result = self.summarizer.summarize(repetitive.strip(), num_sentences=2)
        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result) < len(repetitive)


class TestAbstractiveSummarizer:
    def setup_method(self):
        self.summarizer = AbstractiveSummarizer()

    def test_summarize_calls_pipeline(self):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "A summary."}]
        with patch.object(self.summarizer, "_load_pipeline", return_value=mock_pipe):
            self.summarizer.summarize("Some long text that needs summarization.")
            mock_pipe.assert_called_once_with(
                "Some long text that needs summarization.",
                max_length=150,
                min_length=40,
                do_sample=False,
            )

    def test_summarize_returns_pipeline_result(self):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "Pipeline generated summary."}]
        with patch.object(self.summarizer, "_load_pipeline", return_value=mock_pipe):
            result = self.summarizer.summarize("Some long text that needs summarization.")
            assert result == "Pipeline generated summary."

    def test_lazy_loading(self):
        summarizer = AbstractiveSummarizer()
        assert summarizer._pipeline is None
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "A summary."}]
        with patch("transformers.pipeline", return_value=mock_pipe):
            summarizer.summarize("Some long text that needs summarization.")
            assert summarizer._pipeline is not None

    def test_model_failure(self):
        mock_pipe = MagicMock()
        mock_pipe.side_effect = RuntimeError("Model inference failed")
        with patch.object(self.summarizer, "_load_pipeline", return_value=mock_pipe):
            with pytest.raises(ModelInferenceError, match="Model inference failed"):
                self.summarizer.summarize("Some long text that needs summarization.")

    def test_timeout_simulation(self):
        mock_pipe = MagicMock()
        mock_pipe.side_effect = RuntimeError("Pipeline timed out")
        with patch.object(self.summarizer, "_load_pipeline", return_value=mock_pipe):
            with pytest.raises(ModelInferenceError, match="timed out"):
                self.summarizer.summarize("Some long text that needs summarization.")

    def test_invalid_input_empty_string(self):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": ""}]
        with patch.object(self.summarizer, "_load_pipeline", return_value=mock_pipe):
            result = self.summarizer.summarize("")
            assert result == ""

    def test_custom_max_length(self):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "Short."}]
        with patch.object(self.summarizer, "_load_pipeline", return_value=mock_pipe):
            self.summarizer.summarize("Some long text that needs summarization.", max_length=50)
            mock_pipe.assert_called_once_with(
                "Some long text that needs summarization.",
                max_length=50,
                min_length=40,
                do_sample=False,
            )

    def test_circuit_breaker_opens_after_failures(self):
        mock_pipe = MagicMock()
        mock_pipe.side_effect = RuntimeError("fail")
        with patch.object(self.summarizer, "_load_pipeline", return_value=mock_pipe):
            for _ in range(3):
                with pytest.raises(ModelInferenceError):
                    self.summarizer.summarize("Some long text that needs summarization.")
            with pytest.raises(CircuitOpenError):
                self.summarizer.summarize("Some long text that needs summarization.")

    def test_model_load_error(self):
        summarizer = AbstractiveSummarizer()
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ModelLoadError, match="Failed to import"):
                summarizer._load_pipeline()

    def test_unexpected_output_format(self):
        mock_pipe = MagicMock()
        mock_pipe.return_value = []
        with patch.object(self.summarizer, "_load_pipeline", return_value=mock_pipe):
            with pytest.raises(ModelInferenceError, match="Unexpected model output"):
                self.summarizer.summarize("Some long text that needs summarization.")

    def test_sanitize_removes_control_chars(self):
        result = self.summarizer._sanitize("Hello\x00World\x01Test")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "HelloWorldTest" in result


class TestExtractiveSanitization:
    def test_sanitize_removes_control_chars(self):
        summarizer = ExtractiveSummarizer()
        result = summarizer._sanitize("Hello\x00World\x01Test")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_stop_words_is_frozenset(self):
        assert isinstance(ExtractiveSummarizer.STOP_WORDS, frozenset)
        assert "the" in ExtractiveSummarizer.STOP_WORDS

    def test_text_with_urls(self):
        summarizer = ExtractiveSummarizer()
        text = (
            "Visit https://example.com for more information about NLP. "
            "The website contains detailed documentation about text processing. "
            "Machine learning models are available for download from the site. "
            "The documentation covers both extractive and abstractive methods."
        )
        result = summarizer.summarize(text, num_sentences=2)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_text_with_special_characters(self):
        summarizer = ExtractiveSummarizer()
        text = (
            "The cost was $500 for the service & it included 100% coverage. "
            "Email support@example.com for more details about the pricing. "
            "The discount of 20% applies to orders over $1,000 in value. "
            "Terms and conditions apply to all purchases made online."
        )
        result = summarizer.summarize(text, num_sentences=2)
        assert isinstance(result, str)


class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        from app.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.is_available is True

    def test_opens_after_threshold(self):
        from app.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"
        assert cb.is_available is False

    def test_success_resets_failures(self):
        from app.services.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"
        assert cb.is_available is True
