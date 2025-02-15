import joblib
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, BertForSequenceClassification, BertTokenizer
from concurrent.futures import ThreadPoolExecutor
from transformers import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langdetect import detect
import time

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model 1: Spam Classifier (Optimized with Joblib)
model1 = joblib.load("model1/model1.pkl")
vectorizer1 = joblib.load("model1/vectorizer_model1.pkl")


def classify_spam(text):
    text_transformed = vectorizer1.transform([text])
    return model1.predict(text_transformed)[0]  # 1 for Spam, 0 for Not Spam


# Load Model 2: Hate Speech Detection (LSTM)
with open("model2/final_vocab.pkl", "rb") as f:
    final_vocab = pickle.load(f)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers=1, bidirectional=False, dropout=0.5, pad_idx=0):
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
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]),
                           dim=1) if self.lstm.bidirectional else hidden[-1, :, :]
        return self.fc(self.dropout(hidden))


model2 = LSTMClassifier(len(final_vocab), 128, 128, 3,
                        1, True, 0.5, final_vocab['<PAD>']).to(device)
model2.load_state_dict(torch.load("model2/model2.pt"))
model2.eval()


def predict_hate_speech(text):
    tokens = text.lower().split()
    indices = [final_vocab.get(token, final_vocab['<UNK>'])
               for token in tokens]
    seq_tensor = torch.tensor(
        indices, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(indices)]).to(device)
    with torch.no_grad():
        return F.softmax(model2(seq_tensor, lengths), dim=1).squeeze(0).cpu().numpy()


# Load Model 3: Sarcasm Detection (DistilBERT)
model3 = DistilBertForSequenceClassification.from_pretrained(
    "model3/model3").to(device)
tokenizer3 = DistilBertTokenizerFast.from_pretrained("model3/model3_tokenizer")
model3.eval()


def predict_sarcasm(text):
    inputs = tokenizer3(text, padding=True, truncation=True,
                        return_tensors="pt").to(device)
    with torch.no_grad():
        return F.softmax(model3(**inputs).logits, dim=1).squeeze(0).cpu().numpy()


# Load Model 4: Sentiment Analysis (BERT)
model4 = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3).to(device)
model4.load_state_dict(torch.load("model4/model4.pth"))
model4.eval()
tokenizer4 = BertTokenizer.from_pretrained("bert-base-uncased")


def predict_sentiment(text):
    encoding = tokenizer4(text, padding="max_length", truncation=True,
                          max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        return F.softmax(model4(**encoding).logits, dim=1).squeeze(0).cpu().numpy()

# Load Model 5: Multilingual Sentiment Analysis


model_path = "model5/model5_phase2/"
model5 = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer_model5 = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model5.to(device)


def predict_multilingual_sentiment(text):
    model5.eval()
    inputs = tokenizer_model5(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model5(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

# 🚀 Optimized Final Classification Function


def classify_comment(text):
    if classify_spam(text) == 1:
        return "Spam"

    if detect(text) != "en":
        sentiment_class = predict_multilingual_sentiment(text)
        return ["Negative", "Neutral", "Positive"][sentiment_class]

    # Run hate speech, sarcasm, and sentiment in parallel
    with ThreadPoolExecutor() as executor:
        hate_future = executor.submit(predict_hate_speech, text)
        sarcasm_future = executor.submit(predict_sarcasm, text)
        sentiment_future = executor.submit(predict_sentiment, text)

        hate_speech_probs = hate_future.result()
        sarcasm_probs = sarcasm_future.result()
        sentiment_probs = sentiment_future.result()

    # Apply weighting adjustments
    w_pos, w_neu, w_neg = 1, 1, 1
    if hate_speech_probs[0] > 0.7:
        w_neg += 1.2
        w_pos -= 0.5
    if hate_speech_probs[1] > 0.7:
        w_neg += 0.5
        w_pos -= 0.3
    if sarcasm_probs[1] > 0.7:
        w_pos *= 0.5
        w_neu *= 1.2
        w_neg *= 1.1

    # Compute final scores
    sentiment_scores = [
        sentiment_probs[0] * w_neg,
        sentiment_probs[1] * w_neu,
        sentiment_probs[2] * w_pos
    ]

    # Final decision
    return ["Negative", "Neutral", "Positive"][sentiment_scores.index(max(sentiment_scores))]


# Example Usage
if __name__ == "__main__":

    comments = [
        "Very good movie. I enjoyed it a lot.",
        "Me encantó este libro, es maravilloso.",
        "C'était un film incroyable, je le recommande vivement !",
        "Das Essen war köstlich, ich komme auf jeden Fall wieder!",
        "Che bella giornata, mi sento così felice!",
        "素晴らしい一日でした！"
        "I think this is a spam message.",
        "Creo que este es un mensaje de spam.",
        "Je pense que c'est un message indésirable.",
        "Ich glaube, das ist eine Spam-Nachricht.",
        "Penso che questo sia un messaggio di spam.",
        "これはスパムメッセージだと思います。"
        "This is a terrible product. Do not buy it!",
        "¡Este producto es terrible! ¡No lo compres!",
        "C'est un produit terrible. Ne l'achetez pas !",
        "Dieses Produkt ist schrecklich. Nicht kaufen!",
        "Questo prodotto è orribile. Non comprarlo!",
        "これはひどい商品です。買わないでください！"
        "You are an idiot!",
        "¡Eres un idiota!",
        "Tu es un idiot!",
        "Du bist ein Idiot!",
        "Sei un idiota!"
        "あなたはバカです！"
    ]

    for comment in comments:
        start_time = time.time()
        classification = classify_comment(comment)
        end_time = time.time()
        print(f"Comment: {comment}")
        print(f"Classification: {classification}")
        print(f"Time taken: {end_time - start_time:.4f} seconds\n")
