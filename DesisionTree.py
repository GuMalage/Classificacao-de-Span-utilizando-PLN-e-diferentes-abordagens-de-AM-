import nltk
import kagglehub
import re
import torch
from kagglehub import KaggleDatasetAdapter
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords

set_english_stopwords = set(stopwords.words('english'))

file_path = "twitter_training.csv"

file_path = "SPAM text message 20170820 - Data.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "team-ai/spam-text-message-classification",
    file_path,
)

texts = df["Message"][:5157].astype(str).tolist()
labels = df["Category"][:5157].values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

def removeStopWords(tokens, stopwords_set):
    return [t for t in tokens if t not in stopwords_set]

def cleanText(text):
    return re.sub(r"[^A-Za-zÀ-ÿ ]", "", text)

def toLowerCase(text):
    return text.lower()

corpi_preprocessed = []
for text in texts:
    cleaned = cleanText(text)
    lowered = toLowerCase(cleaned)
    tokens = nltk.word_tokenize(lowered)
    tokens = removeStopWords(tokens, set_english_stopwords)
    corpi_preprocessed.append(" ".join(tokens))

inputs_texts = bert_tokenizer(
    corpi_preprocessed,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

with torch.no_grad():
    output_from_model = bert_model(**inputs_texts)

cls_embeddings = output_from_model.last_hidden_state[:, 0, :].cpu().numpy()

scaler = StandardScaler()
cls_embeddings = scaler.fit_transform(cls_embeddings)

X_train, X_test, y_train, y_test = train_test_split(
    cls_embeddings,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

desision_tree_model = DecisionTreeClassifier()
desision_tree_model.fit(X_train, y_train)

pred = desision_tree_model.predict(X_test)

print("Acurácea:", accuracy_score(y_test, pred))

