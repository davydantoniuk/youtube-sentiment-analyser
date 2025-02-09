import joblib
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, BertForSequenceClassification, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model 1: Spam Classifier
model1 = joblib.load("model1/model1.pkl")
vectorizer1 = joblib.load("model1/vectorizer1.pkl")


def classify_spam(text, model, vectorizer):
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)[0]
    return prediction  # 1 for Spam, 0 for Not Spam


# Load Model 2: Hate Speech Detection
with open("model2/final_vocab.pkl", "rb") as f:
    final_vocab = pickle.load(f)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 n_layers=1, bidirectional=False, dropout=0.5, pad_idx=0):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout if n_layers > 1 else 0)

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (hidden, _) = self.lstm(packed_embedded)

        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits


model2 = LSTMClassifier(
    vocab_size=len(final_vocab),
    embed_dim=128,
    hidden_dim=128,
    output_dim=3,
    n_layers=1,
    bidirectional=True,
    dropout=0.5,
    pad_idx=final_vocab['<PAD>']
)
model2.load_state_dict(torch.load("model2/model2.pt"))
model2.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model2.eval()


def predict_hate_speech(model, sentence, vocab, device):
    def tokenize(text):
        return text.split()
    tokens = tokenize(sentence.lower())
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    seq_tensor = torch.tensor(
        indices, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(indices)]).to(device)
    with torch.no_grad():
        output = model(seq_tensor, lengths)
        probabilities = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
    return probabilities  # [hate_speech_prob, offensive_prob, neither_prob]


# Load Model 3: Sarcasm Detection
model3 = DistilBertForSequenceClassification.from_pretrained(
    "model3/model3").to(device)
tokenizer3 = DistilBertTokenizerFast.from_pretrained("model3/model3_tokenizer")
model3.eval()


def predict_sarcasm(text, model, tokenizer, device):
    inputs = tokenizer(text, padding=True, truncation=True,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
        probabilities = F.softmax(
            output.logits, dim=1).squeeze(0).cpu().numpy()
    return probabilities  # [non_sarcastic_prob, sarcastic_prob]


# Load Model 4: Sentiment Analysis
model4 = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3)
model4.load_state_dict(torch.load("model4/model4.pth"))
model4.to(device)
model4.eval()
tokenizer4 = BertTokenizer.from_pretrained("bert-base-uncased")


def predict_sentiment(text, model, tokenizer, device):
    encoding = tokenizer(text, padding="max_length", truncation=True,
                         max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**encoding)
        probabilities = F.softmax(
            output.logits, dim=1).squeeze(0).cpu().numpy()
    return probabilities  # [negative_prob, neutral_prob, positive_prob]

# Final Decision Function


def classify_comment(text, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Check if it's spam
    if classify_spam(text, model1, vectorizer1) == 1:
        return "Spam"

    # Step 2: Get hate speech probabilities
    hate_speech_probs = predict_hate_speech(model2, text, vocab, device)

    # Step 3: Get sarcasm probabilities
    sarcasm_probs = predict_sarcasm(text, model3, tokenizer3, device)

    # Step 4: Get sentiment probabilities
    sentiment_probs = predict_sentiment(text, model4, tokenizer4, device)

    # Apply weighting adjustments
    w_pos, w_neu, w_neg = 1, 1, 1
    if hate_speech_probs[0] > 0.7:  # High probability of hate speech
        w_neg += 1.2
        w_pos -= 0.5
    if hate_speech_probs[1] > 0.7:  # High probability of offensive language
        w_neg += 0.5
        w_pos -= 0.3
    if sarcasm_probs[1] > 0.7:  # High probability of sarcasm
        w_pos *= 0.5
        w_neu *= 1.2
        w_neg *= 1.1

    # Compute final scores
    sentiment_scores = [
        sentiment_probs[0] * w_neg,
        sentiment_probs[1] * w_neu,
        sentiment_probs[2] * w_pos
    ]

    # Final decision based on highest weighted probability
    sentiment_classes = ["Negative", "Neutral", "Positive"]
    return sentiment_classes[sentiment_scores.index(max(sentiment_scores))]


# Example Usage
if __name__ == "__main__":
    vocab = final_vocab
    test_comment = "Very good movie. I enjoyed it a lot."
    print(classify_comment(test_comment, vocab))
