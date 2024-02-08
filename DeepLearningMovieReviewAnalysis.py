# Importing necessary libraries
import nltk
from nltk.tokenize import word_tokenize
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import movie_reviews
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim

# Downloading NLTK resources
print("Downloading NLTK resources...")
nltk.download('movie_reviews')
nltk.download('wordnet')
nltk.download('punkt')
print("NLTK resources downloaded successfully.")

# Loading movie reviews dataset from NLTK
print("Loading movie reviews dataset...")
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
print("Movie reviews dataset loaded successfully.")

# Shuffle the documents
print("Shuffling the documents...")
random.shuffle(documents)
print("Documents shuffled successfully.")

# Lemmatization for preprocessing optimization
print("Performing lemmatization for preprocessing optimization...")
lemmatizer = WordNetLemmatizer()
documents_lemmatized = [([lemmatizer.lemmatize(w.lower()) for w in d], c) for (d, c) in documents]
print("Lemmatization completed successfully.")

# Split the dataset into features and labels
print("Splitting the dataset into features and labels...")
X = [' '.join(d) for (d, _) in documents_lemmatized]
y = [c for (_, c) in documents_lemmatized]
print("Dataset split into features and labels successfully.")

# Encode labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder()  # Remove sparse=False argument
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_encoded = onehot_encoder.fit_transform(integer_encoded)

# Convert one-hot encoded labels to dense arrays
y_encoded_dense = y_encoded.toarray()

# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("Dataset split into training and testing sets successfully.")

# Convert sparse matrix to dense array
y_train_dense = y_train.toarray()
y_test_dense = y_test.toarray()

# Convert labels to PyTorch tensors
y_train = torch.FloatTensor(y_train_dense)
y_test = torch.FloatTensor(y_test_dense)

# Tokenization using NLTK
print("Tokenizing the text data...")
X_train_tokens = [word_tokenize(text) for text in X_train]
X_test_tokens = [word_tokenize(text) for text in X_test]
print("Text data tokenized successfully.")

# Training Word2Vec model
print("Training Word2Vec model...")
word2vec_model = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5, min_count=1, workers=4)
print("Word2Vec model trained successfully.")

# Creating word embeddings matrix
print("Creating word embeddings matrix...")
embedding_matrix = np.zeros((len(word2vec_model.wv) + 1, word2vec_model.vector_size))
for word, i in word2vec_model.wv.key_to_index.items():
    embedding_matrix[i] = word2vec_model.wv[word]
print("Word embeddings matrix created successfully.")

# Convert tokens to numerical indices
X_train_indices = [[word2vec_model.wv.key_to_index[word] for word in seq if word in word2vec_model.wv.key_to_index]
                   for seq in X_train_tokens]
X_test_indices = [[word2vec_model.wv.key_to_index[word] for word in seq if word in word2vec_model.wv.key_to_index]
                  for seq in X_test_tokens]

# Padding sequences
print("Padding sequences...")
max_sequence_length = max(len(sequence) for sequence in X_train_indices + X_test_indices)
X_train_padded = np.array([seq + [0]*(max_sequence_length-len(seq)) for seq in X_train_indices])
X_test_padded = np.array([seq + [0]*(max_sequence_length-len(seq)) for seq in X_test_indices])
print("Sequences padded successfully.")

# Convert NumPy array to PyTorch tensor
X_train_tensor = torch.LongTensor(X_train_padded)

# Define the LSTMModel class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, embedding_matrix):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

# Instantiate the model
input_size = embedding_matrix.shape[1]  # Embedding dimension
hidden_size = 128
num_layers = 1
output_size = len(label_encoder.classes_)
model = LSTMModel(input_size, hidden_size, num_layers, output_size, embedding_matrix)

# Define loss function (criterion)
criterion = nn.CrossEntropyLoss()

# Instantiate the optimizer
optimizer = optim.Adam(model.parameters())

# Training the model
print("Training the model...")
num_epochs = 10
batch_size = 32  # Adjust the batch size according to your memory constraints
num_batches = len(X_train_padded) // batch_size

for epoch in range(num_epochs):
    total_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs}:")
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        optimizer.zero_grad()
        batch_X = torch.LongTensor(X_train_padded[start_idx:end_idx])
        batch_y = torch.tensor(y_train[start_idx:end_idx], dtype=torch.int64)  # Explicitly specify dtype
        outputs = model(batch_X)
        loss = criterion(outputs, torch.argmax(batch_y, dim=1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")

    average_loss = total_loss / num_batches
    print(f"Average Loss for Epoch {epoch+1}: {average_loss:.4f}")

print("Model trained successfully.")

# Evaluate the model
print("Evaluating the model...")
with torch.no_grad():
    outputs = model(torch.LongTensor(X_test_padded))  # Use X_test_padded here
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == torch.argmax(y_test, dim=1)).sum().item() / len(y_test)
    print(f'Test Accuracy: {accuracy:.4f}')

# Visualize Confusion Matrix
print("Visualizing confusion matrix...")
plt.figure(figsize=(8, 6))
cm = confusion_matrix(np.argmax(y_test.numpy(), axis=1), predicted.numpy())
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()