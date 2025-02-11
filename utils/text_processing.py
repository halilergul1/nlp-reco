import jpype
import os
import string
import re
from jpype import JClass, JString
from langdetect import detect
import src.config as config


class TurkishTextProcessor:
    """
    Handles Turkish text normalization, tokenization, and lemmatization using Zemberek.
    """

    def __init__(self):
        """Initializes Zemberek NLP tools and loads stopwords."""
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[config.ZEMBEREK_PATH])

        self.TurkishTokenizer = JClass("zemberek.tokenization.TurkishTokenizer")
        self.TurkishMorphology = JClass("zemberek.morphology.TurkishMorphology")
        self.TurkishSentenceNormalizer = JClass("zemberek.normalization.TurkishSentenceNormalizer")
        self.Paths = JClass("java.nio.file.Paths")

        self.morphology = self.TurkishMorphology.createWithDefaults()
        self.tokenizer = self.TurkishTokenizer.DEFAULT
        self.normalizer = self.TurkishSentenceNormalizer(
            self.TurkishMorphology.createWithDefaults(),
            self.Paths.get(str(os.path.join(config.DATA_PATH, "normalization"))),
            self.Paths.get(str(os.path.join(config.DATA_PATH, "lm", "lm.2gram.slm")))
        )

        self.stopwords = self.load_stopwords()

    def load_stopwords(self):
        """Loads Turkish stopwords from a file."""
        stopword_file = config.STOPWORDS_FILE
        if os.path.exists(stopword_file):
            with open(stopword_file, "r", encoding="utf-8") as file:
                return set(line.strip() for line in file)
        return set()

    def clean_html_tags(self, text):
        """
        Cleans unwanted HTML tags and special characters from text.

        Args:
            text (str): Input text containing HTML tags.

        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str):
            return text  # Return unchanged if it's not a string
        
        # Remove HTML tags and unnecessary symbols (I realized this in the data !)
        text = re.sub(r'<br\s*/?>|</?p>', ' ', text)
        text = text.replace('/', ' ').replace('\n', ' ')
        text = re.sub(r'\bmi\b', '', text)

        return text.strip()

    def normalize_text(self, text):
        """
        Normalizes text (handling Turkish-specific transformations).
        It detects the language and normalizes Turkish words only because it converts English words to UNK.

        Args:
            text (str): Input text.

        Returns:
            str: Normalized text.
        """
        words = text.split()
        normalized_words = []

        for word in words:
            try:
                if detect(word) == "tr":  # Only normalize Turkish words
                    normalized_words.append(str(self.normalizer.normalize(JString(word))))
                else:
                    normalized_words.append(word)  # Keep English words unchanged
            except:
                normalized_words.append(word)  # If detection fails, keep the word unchanged

        return " ".join(normalized_words)

    def remove_punctuation(self, text):
        """Removes punctuation from text."""
        return "".join([char for char in text if char not in string.punctuation])

    def remove_numbers(self, text):
        """Removes numeric characters from text."""
        return "".join([char for char in text if not char.isdigit()])

    def tokenize(self, text):
        """Tokenizes text using Zemberek."""
        return [str(token) for token in self.tokenizer.tokenizeToStrings(JString(text))]

    def remove_stopwords(self, tokens):
        """Removes stopwords from a tokenized list of words."""
        return [word for word in tokens if word not in self.stopwords]

    def analyze_words(self, tokens):
        """
        Morphologically analyzes each word using Zemberek.
        Only applies analysis if the word is Turkish.

        Args:
            tokens (list): Tokenized words.

        Returns:
            list: Morphologically analyzed tokens.
        """
        analyzed_tokens = []
        for word in tokens:
            try:
                if detect(word) == "tr":
                    analysis = self.morphology.analyzeAndDisambiguate(JString(word)).bestAnalysis()
                    analyzed_tokens.append(analysis[0])
                else:
                    analyzed_tokens.append(word)  # Keep English words unchanged
            except:
                analyzed_tokens.append(word)  # If detection fails, keep word unchanged

        return analyzed_tokens

    def lemmatize(self, analysis_list):
        """
        Extracts lemmas from analyzed tokens.

        Args:
            analysis_list (list): List of morphological analyses.

        Returns:
            list: Lemmatized words.
        """
        lemmatized_words = []
        for analysis in analysis_list:
            if isinstance(analysis, str):  # If it's an English word, keep it unchanged
                lemmatized_words.append(analysis)
            else:
                lemmatized_words.append(str(analysis.getDictionaryItem().lemma))

        return lemmatized_words

    def process_text(self, text):
        """
        Applies full Turkish text processing pipeline:
        1. Normalizes text
        2. Removes punctuation and numbers
        3. Tokenizes
        4. Removes stopwords
        5. Analyzes morphology
        6. Lemmatizes
        7. Returns processed text
        
        Args:
            text (str): Raw input text.
        
        Returns:
            str: Cleaned and processed text.
        """
        text = self.clean_html_tags(text)
        text = self.normalize_text(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)

        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        analyzed = self.analyze_words(tokens)
        lemmatized = self.lemmatize(analyzed)

        return " ".join(lemmatized)

def process_hotel_descriptions(hotel_desc):
    """
    Cleans and preprocesses hotel descriptions.
    
    Args:
        hotel_desc (DataFrame): Raw hotel description dataset.
    
    Returns:
        hotel_desc (DataFrame): Processed hotel descriptions.
    """

    processor = TurkishTextProcessor()
    
    # Apply HTML cleaning
    hotel_desc["cleaned_description"] = hotel_desc["description_text"].apply(processor.clean_html_tags)
    
    # Apply full NLP pipeline (lemmatization, normalization, stopword removal)
    hotel_desc["lemmatized_text"] = hotel_desc["cleaned_description"].apply(lambda x: processor.process_text(x))

    return hotel_desc

def save_processed_descriptions(hotel_desc):
    """
    Saves processed hotel descriptions to CSV.
    
    Args:
        hotel_desc (DataFrame): The processed descriptions.
    """
    hotel_desc.to_csv(f"{config.DATA_PATH}/hotel_desc_lemmatized.csv", index=False)
    print(" Processed hotel descriptions saved as 'hotel_desc_lemmatized.csv'.")
