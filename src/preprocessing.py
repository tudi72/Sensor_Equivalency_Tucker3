from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string

from utils.utils import generate_new_df_with_transformed_column


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, text_column='text'):
        self.text_column = text_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Tokenize the text in the specified column."""

        X_transformed = generate_new_df_with_transformed_column(X, self.text_column, word_tokenize)

        return X_transformed


class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, use_lemmatizer=True, text_column='text'):
        self.text_column = text_column
        self.use_lemmatizer = use_lemmatizer
        self.lemmatizer = WordNetLemmatizer() if use_lemmatizer else None
        self.stemmer = PorterStemmer() if not use_lemmatizer else None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Normalize the tokenized text: lowercasing, lemmatization or stemming."""

        def normalize(tokens):
            tokens = [token.lower() for token in tokens if token.isalpha()]
            if self.lemmatizer:
                return [self.lemmatizer.lemmatize(token) for token in tokens]
            elif self.stemmer:
                return [self.stemmer.stem(token) for token in tokens]
            return tokens

        X_transformed = generate_new_df_with_transformed_column(X, self.text_column, normalize)

        return X_transformed


class TokenFilter(BaseEstimator, TransformerMixin):
    def __init__(self, language="english", text_column="text"):
        self.text_column = text_column
        self.stop_words = set(stopwords.words(language))
        self.punctuation = set(string.punctuation)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Filter stop words and punctuation."""

        def filter_tokens(tokens):
            return [token for token in tokens if token not in self.stop_words and token not in self.punctuation]

        X_transformed = generate_new_df_with_transformed_column(X, self.text_column, filter_tokens)

        return X_transformed