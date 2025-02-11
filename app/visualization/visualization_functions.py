from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib
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
