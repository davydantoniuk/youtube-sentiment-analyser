import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
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
    documents,
    ngram_range=(1, 1),
    min_word_length=8,
    stopwords_list=None,
    top_n=None
):
    """
    Given a list of documents (strings), this function:
      - Uses TF-IDF to compute term frequencies
      - Optionally removes custom stopwords (multilingual or otherwise)
      - Penalizes words shorter than `min_word_length` by multiplying
        their TF-IDF score by (len(word) / min_word_length).
      - Returns a dict of {term -> adjusted TF-IDF score}, optionally limited to top_n.

    :param documents: list of text strings
    :param ngram_range: (lower, upper) for n-grams, e.g. (1,1) for single words, (2,3) for bigrams+trigrams, etc.
    :param min_word_length: words shorter than this get penalized
    :param stopwords_list: pass a list of stopwords if desired
    :param top_n: if not None, returns only the top_n by score
    :return: dict {term: float_score}
    """
    if stopwords_list is None:
        stopwords_list = []

    vectorizer = TfidfVectorizer(
        stop_words=stopwords_list,
        ngram_range=ngram_range,
        token_pattern=r"\b\w+\b",
    )

    X = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1

    freq_dict = {}
    for word, score in zip(feature_names, scores):
        if len(word) < min_word_length:
            ratio = len(word) / float(min_word_length)
            # Apply a stronger penalty for shorter words:
            adjusted_score = score * (ratio ** 2)
            freq_dict[word] = adjusted_score
        else:
            freq_dict[word] = score

    if top_n is not None:
        sorted_items = sorted(
            freq_dict.items(), key=lambda x: x[1], reverse=True)
        freq_dict = dict(sorted_items[:top_n])

    return freq_dict


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

def create_word_cloud(comments, top_n=200, min_word_length=8):
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
        ngram_range=(1, 1),
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


# 5) WORD FREQUENCY BAR PLOTS (using TF-IDF, short-word penalty)

def create_word_frequency_barplots(positive_comments, neutral_comments, negative_comments, top_n=10):
    """
    Creates bar plots showing the most important words in positive, neutral, and negative comments,
    based on TF-IDF scoring plus a penalty for words under 6 letters.
    Returns the plot as a base64-encoded PNG string.
    """

    # Extract top TF-IDF frequencies for each sentiment
    pos_freqs = compute_tfidf_frequencies(
        documents=positive_comments,
        ngram_range=(1, 1),
        min_word_length=8,
        stopwords_list=MULTI_STOPWORDS,
        top_n=top_n
    )
    neg_freqs = compute_tfidf_frequencies(
        documents=negative_comments,
        ngram_range=(1, 1),
        min_word_length=8,
        stopwords_list=MULTI_STOPWORDS,
        top_n=top_n
    )
    neu_freqs = compute_tfidf_frequencies(
        documents=neutral_comments,
        ngram_range=(1, 1),
        min_word_length=8,
        stopwords_list=MULTI_STOPWORDS,
        top_n=top_n
    )

    # Convert dictionaries to sorted lists [(word, freq), ...]
    def sort_dict(d):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)

    pos_common = sort_dict(pos_freqs)
    neg_common = sort_dict(neg_freqs)
    neu_common = sort_dict(neu_freqs)

    # --- Create figure and GridSpec layout
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.5])

    # Subplots
    ax_pos = fig.add_subplot(gs[0, 0])  # Positive
    ax_neg = fig.add_subplot(gs[0, 1])  # Negative
    ax_neu = fig.add_subplot(gs[1, :])  # Neutral

    color_map = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

    # Plot Positive
    if pos_common:
        words, scores = zip(*pos_common)
        ax_pos.barh(words, scores, color=color_map['Positive'])
        ax_pos.set_title("Positive Words (TF-IDF)")
        ax_pos.set_yticks(range(len(words)))
        ax_pos.set_yticklabels(words, rotation=0, ha='right')
    else:
        ax_pos.axis('off')

    # Plot Negative
    if neg_common:
        words, scores = zip(*neg_common)
        ax_neg.barh(words, scores, color=color_map['Negative'])
        ax_neg.set_title("Negative Words (TF-IDF)")
        ax_neg.set_yticks(range(len(words)))
        ax_neg.set_yticklabels(words, rotation=0, ha='right')
    else:
        ax_neg.axis('off')

    # Plot Neutral
    if neu_common:
        words, scores = zip(*neu_common)
        ax_neu.barh(words, scores, color=color_map['Neutral'])
        ax_neu.set_title("Neutral Words (TF-IDF)")
        ax_neu.set_yticks(range(len(words)))
        ax_neu.set_yticklabels(words, rotation=0, ha='right')
    else:
        ax_neu.axis('off')

    plt.tight_layout()

    # Encode figure as base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"


# 6) PHRASE FREQUENCY BAR PLOTS (TF-IDF on ngrams)

def create_phrase_frequency_barplots(
    positive_comments,
    neutral_comments,
    negative_comments,
    top_n=10,
    ngram_range=(2, 3)
):
    """
    Creates bar plots of the most important phrases (bigrams/trigrams/etc.)
    in positive, neutral, and negative comments using TF-IDF, with optional short-word penalty.
    Returns the plot as a base64-encoded PNG string.
    """

    pos_phrases = compute_tfidf_frequencies(
        documents=positive_comments,
        ngram_range=ngram_range,
        min_word_length=8,
        stopwords_list=MULTI_STOPWORDS,
        top_n=top_n
    )
    neg_phrases = compute_tfidf_frequencies(
        documents=negative_comments,
        ngram_range=ngram_range,
        min_word_length=8,
        stopwords_list=MULTI_STOPWORDS,
        top_n=top_n
    )
    neu_phrases = compute_tfidf_frequencies(
        documents=neutral_comments,
        ngram_range=ngram_range,
        min_word_length=8,
        stopwords_list=MULTI_STOPWORDS,
        top_n=top_n
    )

    def sort_dict(d):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)

    pos_common = sort_dict(pos_phrases)
    neg_common = sort_dict(neg_phrases)
    neu_common = sort_dict(neu_phrases)

    # Layout
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.5])

    ax_pos = fig.add_subplot(gs[0, 0])
    ax_neg = fig.add_subplot(gs[0, 1])
    ax_neu = fig.add_subplot(gs[1, :])

    color_map = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral': 'blue'
    }

    # Positive
    if pos_common:
        phrases, scores = zip(*pos_common)
        ax_pos.barh(phrases, scores, color=color_map['Positive'])
        ax_pos.set_title("Positive Phrases (TF-IDF)")
        ax_pos.set_yticks(range(len(phrases)))
        ax_pos.set_yticklabels(phrases, rotation=0, ha='right')
    else:
        ax_pos.axis('off')

    # Negative
    if neg_common:
        phrases, scores = zip(*neg_common)
        ax_neg.barh(phrases, scores, color=color_map['Negative'])
        ax_neg.set_title("Negative Phrases (TF-IDF)")
        ax_neg.set_yticks(range(len(phrases)))
        ax_neg.set_yticklabels(phrases, rotation=0, ha='right')
    else:
        ax_neg.axis('off')

    # Neutral
    if neu_common:
        phrases, scores = zip(*neu_common)
        ax_neu.barh(phrases, scores, color=color_map['Neutral'])
        ax_neu.set_title("Neutral Phrases (TF-IDF)")
        ax_neu.set_yticks(range(len(phrases)))
        ax_neu.set_yticklabels(phrases, rotation=0, ha='right')
    else:
        ax_neu.axis('off')

    plt.tight_layout()

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{encoded}"
