from nltk.corpus import stopwords
from collections import Counter
import re
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
from nltk.util import ngrams
matplotlib.use('Agg')


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


def create_word_frequency_barplots(positive_comments, neutral_comments, negative_comments, top_n=10):
    """
    Creates bar plots showing the most common words in positive, neutral, and negative comments.
    Returns the plot as a base64-encoded PNG string.
    """

    def process_text(comments):
        words = []
        for text in comments:
            tokens = re.findall(r'\w+', text.lower())
            tokens = [t for t in tokens if t not in stopwords.words('english')]
            words.extend(tokens)
        return Counter(words).most_common(top_n)

    # Get top words
    pos_common = process_text(positive_comments)
    neg_common = process_text(negative_comments)
    neu_common = process_text(neutral_comments)

    # --- Create a figure and specify a GridSpec with 2 rows and 2 columns ---
    #     We'll let the bottom subplot span both columns (width colspan).
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.5])

    # Top row subplots (two columns)
    ax_pos = fig.add_subplot(gs[0, 0])  # Positive words (top-left)
    ax_neg = fig.add_subplot(gs[0, 1])  # Negative words (top-right)

    # Bottom subplot spanning both columns: [1, :]
    # which means row=1, all columns => the neutral plot will be "centered"
    ax_neu = fig.add_subplot(gs[1, :])  # Neutral words (bottom, centered)

    # Define color mapping for clarity
    color_map = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

    # Plot Positive
    if pos_common:
        words, counts = zip(*pos_common)
        ax_pos.barh(words, counts, color=color_map['Positive'])
        ax_pos.set_title("Positive Words")
        ax_pos.set_yticks(range(len(words)))
        ax_pos.set_yticklabels(words, rotation=0, ha='right')
    else:
        ax_pos.axis('off')

    # Plot Negative
    if neg_common:
        words, counts = zip(*neg_common)
        ax_neg.barh(words, counts, color=color_map['Negative'])
        ax_neg.set_title("Negative Words")
        ax_neg.set_yticks(range(len(words)))
        ax_neg.set_yticklabels(words, rotation=0, ha='right')
    else:
        ax_neg.axis('off')

    # Plot Neutral
    if neu_common:
        words, counts = zip(*neu_common)
        ax_neu.barh(words, counts, color=color_map['Neutral'])
        ax_neu.set_title("Neutral Words")
        ax_neu.set_yticks(range(len(words)))
        ax_neu.set_yticklabels(words, rotation=0, ha='right')
    else:
        ax_neu.axis('off')

    # Final layout
    plt.tight_layout()

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{encoded}"


def create_phrase_frequency_barplots(
    positive_comments,
    neutral_comments,
    negative_comments,
    top_n=10,
    ngram_range=(2, 3)
):
    """
    Creates bar plots showing the most common phrases in positive, neutral,
    and negative comments. Returns the plot as a base64-encoded PNG string.
    """

    def process_text(comments, ngram_range):
        phrases = []
        for text in comments:
            # Tokenize
            tokens = re.findall(r'\w+', text.lower())
            # Remove stopwords
            tokens = [t for t in tokens if t not in stopwords.words('english')]
            # Generate n-grams (e.g. bigrams, trigrams, ...)
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for gram in ngrams(tokens, n):
                    phrases.append(' '.join(gram))
        return Counter(phrases).most_common(top_n)

    # Get top phrase frequencies
    pos_common = process_text(positive_comments, ngram_range)
    neg_common = process_text(negative_comments, ngram_range)
    neu_common = process_text(neutral_comments, ngram_range)

    # Create figure and use GridSpec for layout
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.5])

    # Top row subplots
    ax_pos = fig.add_subplot(gs[0, 0])  # Positive (top-left)
    ax_neg = fig.add_subplot(gs[0, 1])  # Negative (top-right)

    # Bottom subplot spans both columns => neutral (centered)
    ax_neu = fig.add_subplot(gs[1, :])

    # Define colors
    color_map = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral':  'blue'
    }

    # Plot Positive
    if pos_common:
        phrases, counts = zip(*pos_common)
        ax_pos.barh(phrases, counts, color=color_map['Positive'])
        ax_pos.set_title("Positive Phrases")
        ax_pos.set_yticks(range(len(phrases)))
        ax_pos.set_yticklabels(phrases, rotation=0, ha='right')
    else:
        ax_pos.axis('off')

    # Plot Negative
    if neg_common:
        phrases, counts = zip(*neg_common)
        ax_neg.barh(phrases, counts, color=color_map['Negative'])
        ax_neg.set_title("Negative Phrases")
        ax_neg.set_yticks(range(len(phrases)))
        ax_neg.set_yticklabels(phrases, rotation=0, ha='right')
    else:
        ax_neg.axis('off')

    # Plot Neutral (bottom)
    if neu_common:
        phrases, counts = zip(*neu_common)
        ax_neu.barh(phrases, counts, color=color_map['Neutral'])
        ax_neu.set_title("Neutral Phrases")
        ax_neu.set_yticks(range(len(phrases)))
        ax_neu.set_yticklabels(phrases, rotation=0, ha='right')
    else:
        ax_neu.axis('off')

    # Tight layout
    plt.tight_layout()

    # Convert plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{encoded}"


def create_word_cloud(comments):
    """
    Creates a word cloud from the given comments and returns the image as a base64 string.
    """
    text = ' '.join(comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=stopwords.words('english')).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"
