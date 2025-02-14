import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from nltk.corpus import stopwords
import base64
import re
from io import BytesIO
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

matplotlib.use('Agg')

nltk.download('stopwords')


# 1) MULTILINGUAL STOPWORDS (Optional)

LANGS = ["english", "polish", "russian", "ukrainian"]
MULTI_STOPWORDS = set()

for lang in LANGS:
    try:
        MULTI_STOPWORDS.update(stopwords.words(lang))
    except OSError:
        pass

MULTI_STOPWORDS = list(MULTI_STOPWORDS)


# 2) CORE TF-IDF + SHORT-WORD PENALTY FUNCTION

def compute_tfidf_frequencies(
    documents, ngram_range=(1, 1), min_word_length=6, stopwords_list=None, top_n=None
):
    """
    Computes TF-IDF frequencies with error handling.
    If the input is too small or contains only stopwords, returns an empty dictionary.
    """
    if stopwords_list is None:
        stopwords_list = []

    try:
        vectorizer = TfidfVectorizer(
            stop_words=stopwords_list, ngram_range=ngram_range, token_pattern=r"\b\w+\b"
        )
        X = vectorizer.fit_transform(documents)

        if X.shape[1] == 0:
            raise ValueError("empty vocabulary")

        feature_names = vectorizer.get_feature_names_out()
        scores = X.sum(axis=0).A1

        freq_dict = {}
        for word, score in zip(feature_names, scores):
            adjusted_score = score * \
                (len(word) / float(min_word_length)
                 ) if len(word) < min_word_length else score
            freq_dict[word] = adjusted_score

        if top_n:
            freq_dict = dict(
                sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])

        return freq_dict

    except (ValueError, NotFittedError):
        # If the vocabulary is empty, return an empty dictionary
        return {}


# 3) SENTIMENT BAR PLOT (unchanged)

def create_sentiment_barplot(positive, neutral, negative):
    """
    Creates a bar plot of the sentiment counts and returns the image as a base64 string.
    """
    labels = ['Positive', 'Neutral', 'Negative']
    values = [positive, neutral, negative]

    plt.figure(figsize=(4, 3))
    plt.bar(labels, values, color=['green', 'blue', 'red'])
    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


# 4) WORD CLOUD USING TF-IDF (with short-word penalty)

def create_word_cloud(comments, top_n=200, min_word_length=6):
    """
    Creates a word cloud from TF-IDF frequencies (across all `comments`).
    We apply short-word penalty and (optionally) multilingual stopwords.
    Then pick the top_n terms for the cloud.

    :param comments: list of text strings
    :param top_n: how many terms to include in the word cloud
    :param min_word_length: short words get penalized
    :return: Base64-encoded PNG
    """
    # 1. Compute TF-IDF frequencies
    freq_dict = compute_tfidf_frequencies(
        documents=comments,
        ngram_range=(1, 1),          # single words
        min_word_length=min_word_length,
        stopwords_list=MULTI_STOPWORDS,
        top_n=top_n
    )

    # 2. Create the word cloud from frequencies
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(freq_dict)

    # 3. Plot + encode as base64
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


# 5) WORD FREQUENCY BAR PLOTS (Fixed Size)


def create_word_frequency_barplots(positive_comments, neutral_comments, negative_comments, top_n=5):
    """
    Creates word frequency plots ensuring bars are properly spaced and all subplots have equal sizes.
    """
    pos_freqs = compute_tfidf_frequencies(
        positive_comments, ngram_range=(1, 1), top_n=top_n)
    neg_freqs = compute_tfidf_frequencies(
        negative_comments, ngram_range=(1, 1), top_n=top_n)
    neu_freqs = compute_tfidf_frequencies(
        neutral_comments, ngram_range=(1, 1), top_n=top_n)

    if not pos_freqs and not neg_freqs and not neu_freqs:
        return None  # Skip plot if all are empty

    def sort_dict(d):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)

    pos_common, neg_common, neu_common = sort_dict(
        pos_freqs), sort_dict(neg_freqs), sort_dict(neu_freqs)

    # Fixed size (same for both plots)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    color_map = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

    def plot_freq(ax, common_words, title, color):
        if common_words:
            words, scores = zip(*common_words)
            y_pos = range(len(words))
            ax.barh(y_pos, scores, color=color)
            ax.set_title(title, fontsize=12)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=10)
            ax.set_ylim(-0.5, top_n - 0.5)  # Ensure same height across plots
            ax.invert_yaxis()  # Highest values on top
        else:
            ax.axis('off')

    plot_freq(axes[0], pos_common, "Positive Words (TF-IDF)",
              color_map['Positive'])
    plot_freq(axes[1], neg_common, "Negative Words (TF-IDF)",
              color_map['Negative'])
    plot_freq(axes[2], neu_common, "Neutral Words (TF-IDF)",
              color_map['Neutral'])

    plt.subplots_adjust(hspace=0.5)  # Equal spacing between subplots
    plt.tight_layout()  # Ensures a cleaner look

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Ensure tight layout
    plt.close()
    buf.seek(0)

    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


# 6) PHRASE FREQUENCY BAR PLOTS (Fixed Size)


def create_phrase_frequency_barplots(positive_comments, neutral_comments, negative_comments, top_n=5, ngram_range=(2, 3)):
    """
    Creates phrase frequency plots ensuring bars are properly spaced and all subplots have equal sizes.
    """
    pos_phrases = compute_tfidf_frequencies(
        positive_comments, ngram_range=ngram_range, top_n=top_n)
    neg_phrases = compute_tfidf_frequencies(
        negative_comments, ngram_range=ngram_range, top_n=top_n)
    neu_phrases = compute_tfidf_frequencies(
        neutral_comments, ngram_range=ngram_range, top_n=top_n)

    if not pos_phrases and not neg_phrases and not neu_phrases:
        return None  # Skip plot if all are empty

    def sort_dict(d):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)

    pos_common, neg_common, neu_common = sort_dict(
        pos_phrases), sort_dict(neg_phrases), sort_dict(neu_phrases)

    # Fixed size (same as word plot)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    color_map = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

    def plot_freq(ax, common_phrases, title, color):
        if common_phrases:
            phrases, scores = zip(*common_phrases)
            y_pos = range(len(phrases))
            ax.barh(y_pos, scores, color=color)
            ax.set_title(title, fontsize=12)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(phrases, fontsize=10)
            ax.set_ylim(-0.5, top_n - 0.5)  # Ensure same height across plots
            ax.invert_yaxis()  # Highest values on top
        else:
            ax.axis('off')

    plot_freq(axes[0], pos_common, "Positive Phrases (TF-IDF)",
              color_map['Positive'])
    plot_freq(axes[1], neg_common, "Negative Phrases (TF-IDF)",
              color_map['Negative'])
    plot_freq(axes[2], neu_common, "Neutral Phrases (TF-IDF)",
              color_map['Neutral'])

    plt.subplots_adjust(hspace=0.5)  # Equal spacing between subplots
    plt.tight_layout()  # Ensures a cleaner look

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Ensure tight layout
    plt.close()
    buf.seek(0)

    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
