import os
import re
import string
from abc import ABC, abstractmethod
from typing import List

import src.config as config


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseTextProcessor(ABC):
    """
    Abstract contract for text processors.

    Concrete subclasses implement language- or library-specific logic.
    The fixed pipeline skeleton (process_text) is defined here so callers
    never need to know which implementation is active.
    """

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        ...

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Language-specific normalization (e.g., Turkish spelling fixes)."""
        ...

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        ...

    @abstractmethod
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter stopwords from a token list."""
        ...

    @abstractmethod
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Reduce tokens to their base/root form."""
        ...

    def process_text(self, text: str) -> str:
        """
        Fixed pipeline: clean → normalize → tokenize → remove stopwords → lemmatize.

        Subclasses implement each stage; this method is never overridden.
        """
        text = self.clean_text(text)
        text = self.normalize(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)


# ---------------------------------------------------------------------------
# Primary implementation — Zemberek (requires Java)
# ---------------------------------------------------------------------------

class TurkishTextProcessor(BaseTextProcessor):
    """
    Turkish text processor backed by Zemberek (via JPype).

    Requires:
      - Java runtime
      - jars/zemberek-full.jar  (configured in config.ZEMBEREK_PATH)
    """

    def __init__(self):
        import jpype
        from jpype import JClass, JString

        if not jpype.isJVMStarted():
            # NOTE: Requires Java 11+ on Apple Silicon (ARM64).
            # Java 8 on ARM64 crashes with a SIGSEGV in jshort_disjoint_arraycopy.
            # Install Java 11+ via: brew install --cask temurin@11
            jpype.startJVM(classpath=[config.ZEMBEREK_PATH])

        self._JString = JString
        self.TurkishTokenizer = JClass("zemberek.tokenization.TurkishTokenizer")
        self.TurkishMorphology = JClass("zemberek.morphology.TurkishMorphology")
        self.TurkishSentenceNormalizer = JClass("zemberek.normalization.TurkishSentenceNormalizer")
        Paths = JClass("java.nio.file.Paths")

        self.morphology = self.TurkishMorphology.createWithDefaults()
        self.tokenizer_obj = self.TurkishTokenizer.DEFAULT
        self.normalizer = self.TurkishSentenceNormalizer(
            self.TurkishMorphology.createWithDefaults(),
            Paths.get(str(os.path.join(config.DATA_PATH, "normalization"))),
            Paths.get(str(os.path.join(config.DATA_PATH, "lm", "lm.2gram.slm")))
        )
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self):
        stopword_file = config.STOPWORDS_FILE
        if os.path.exists(stopword_file):
            with open(stopword_file, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f)
        return set()

    # -- BaseTextProcessor interface --

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        text = re.sub(r'<br\s*/?>|</?p>', ' ', text)
        text = text.replace('/', ' ').replace('\n', ' ')
        text = re.sub(r'\bmi\b', '', text)
        return text.strip()

    def normalize(self, text: str) -> str:
        from langdetect import detect
        words = text.split()
        normalized = []
        for word in words:
            try:
                if detect(word) == "tr":
                    normalized.append(str(self.normalizer.normalize(self._JString(word))))
                else:
                    normalized.append(word)
            except Exception:
                normalized.append(word)
        return " ".join(normalized)

    def tokenize(self, text: str) -> List[str]:
        text = "".join(ch for ch in text if ch not in string.punctuation and not ch.isdigit())
        return [str(tok) for tok in self.tokenizer_obj.tokenizeToStrings(self._JString(text))]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [w for w in tokens if w not in self.stopwords]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        from langdetect import detect
        lemmas = []
        for word in tokens:
            try:
                if detect(word) == "tr":
                    analysis = self.morphology.analyzeAndDisambiguate(self._JString(word)).bestAnalysis()
                    lemmas.append(str(analysis[0].getDictionaryItem().lemma))
                else:
                    lemmas.append(word)
            except Exception:
                lemmas.append(word)
        return lemmas


# ---------------------------------------------------------------------------
# Fallback implementation — no Java, no special deps
# ---------------------------------------------------------------------------

class FallbackTextProcessor(BaseTextProcessor):
    """
    Lightweight text processor that works in any environment (no Java needed).

    Suitable for CI, demos, and contributors who don't have Zemberek installed.
    Lemmatization is an identity operation (tokens returned as-is).
    """

    def __init__(self):
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self):
        stopword_file = config.STOPWORDS_FILE
        if os.path.exists(stopword_file):
            with open(stopword_file, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f)
        return set()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        text = re.sub(r'<br\s*/?>|</?p>', ' ', text)
        text = text.replace('/', ' ').replace('\n', ' ')
        text = re.sub(r'\bmi\b', '', text)
        return text.strip()

    def normalize(self, text: str) -> str:
        return text.lower()

    def tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        return text.split()

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [w for w in tokens if w not in self.stopwords]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        # Identity — no morphological analysis without Zemberek
        return tokens


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_text_processor(processor_type: str = "turkish") -> BaseTextProcessor:
    """
    Factory that returns a text processor by name.

    Args:
        processor_type: "turkish" (Zemberek, requires Java) or "fallback" (regex only).

    Returns:
        An instance of a BaseTextProcessor subclass.
    """
    if processor_type == "turkish":
        return TurkishTextProcessor()
    elif processor_type == "fallback":
        return FallbackTextProcessor()
    else:
        raise ValueError(f"Unknown processor type '{processor_type}'. Choose 'turkish' or 'fallback'.")


# ---------------------------------------------------------------------------
# Module-level pipeline function (called by main.py)
# ---------------------------------------------------------------------------

def process_hotel_descriptions(hotel_desc, processor: BaseTextProcessor = None):
    """
    Cleans and preprocesses hotel descriptions.

    Args:
        hotel_desc (DataFrame): Raw hotel description dataset.
        processor (BaseTextProcessor, optional): Processor instance to use.
            Defaults to the type configured in config.PROCESSOR_TYPE.

    Returns:
        hotel_desc (DataFrame): Processed hotel descriptions with
            'cleaned_description' and 'lemmatized_text' columns.
    """
    if processor is None:
        processor = get_text_processor(config.PROCESSOR_TYPE)

    hotel_desc["cleaned_description"] = hotel_desc["description_text"].apply(processor.clean_text)
    hotel_desc["lemmatized_text"] = hotel_desc["cleaned_description"].apply(processor.process_text)

    return hotel_desc


def save_processed_descriptions(hotel_desc):
    """
    Saves processed hotel descriptions to CSV.

    Args:
        hotel_desc (DataFrame): The processed descriptions.
    """
    hotel_desc.to_csv(f"{config.DATA_PATH}/hotel_desc_lemmatized.csv", index=False)
    print(" Processed hotel descriptions saved as 'hotel_desc_lemmatized.csv'.")
