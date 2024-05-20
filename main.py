import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader

# Ensure NLTK stopwords dataset is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters for the LSTM model
EMBEDDING_DIM = 30
HIDDEN_DIM = 128
OUTPUT_DIM = 1
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 16

def read_text_files(directory):
    texts = []
    for category in ['ham', 'spam']:
        category_path = os.path.join(directory, category)
        if os.path.exists(category_path):
            for filename in os.listdir(category_path):
                filepath = os.path.join(category_path, filename)
                with open(filepath, "r", encoding="latin-1") as file:
                    text = file.read()
                    texts.append((text, 0 if category == 'ham' else 1))
    return texts

def preprocess_text(texts):
    print("Preprocessing texts...")
    stop_words = set(stopwords.words('english'))
    preprocessed_texts = []
    
    for text, label in texts:
        text = text.lower()  # Lowercase conversion
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
        text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])  # Remove stopwords
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        tokens = word_tokenize(text)  # Tokenize
        preprocessed_texts.append((tokens, label))
    
    print(f"First preprocessed text sample: {preprocessed_texts[0]}")
    return preprocessed_texts

def text_to_sequence(texts):
    print("Converting texts to sequences...")
    vocab = set()
    for text, _ in texts:
        vocab.update(text)
    word_to_index = {word: idx + 1 for idx, word in enumerate(vocab)}
    sequences = [([word_to_index[word] for word in text], label) for text, label in texts]
    print(f"Vocabulary size: {len(vocab)}")
    print(f"First sequence sample: {sequences[0]}")
    return sequences, word_to_index

def load_data(data_dir):
    print(f"Loading data from directory: {data_dir}")
    data = []
    for enron_folder in os.listdir(data_dir):
        enron_path = os.path.join(data_dir, enron_folder)
        if os.path.isdir(enron_path):
            data.extend(read_text_files(enron_path))
    texts, labels = zip(*data)
    print(f"Number of texts loaded: {len(texts)}")
    return list(texts), list(labels)

class TextDataset(Dataset):
    def __init__(self, texts, labels, max_length):
        self.texts = [self.pad_sequence(text, max_length) for text in texts]
        self.labels = labels
        print(f"Dataset created with {len(self.texts)} samples.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

    def pad_sequence(self, sequence, max_length):
        padded_sequence = np.zeros(max_length, dtype=int)
        padded_sequence[:len(sequence)] = sequence
        return padded_sequence

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, vocab_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return self.sigmoid(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_model", action="store_true", help="Train the model if this flag is provided.")
    args = parser.parse_args()

    # Load data
    data_dir = "path/to/your/data"  # Update this with the correct path to your data directory
    texts, labels = load_data(data_dir)

    # Preprocess and tokenize texts
    texts_cleaned = preprocess_text(list(zip(texts, labels)))
    sequences, vocab = text_to_sequence(texts_cleaned)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split([seq for seq, label in sequences], [label for seq, label in sequences], test_size=0.2, stratify=labels, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # Create Dataset and DataLoader
    max_length = max(len(seq) for seq in X_train) + 10
    train_dataset = TextDataset(X_train, y_train, max_length)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TextDataset(X_test, y_test, max_length)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create the LSTM model
    model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, len(vocab) + 1).to(device)  # +1 for padding index
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if args.train_model:
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        # Training loop
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            for texts, labels in train_loader:
                texts, labels = texts.to(device), labels.to(device)
                
                print(f"Training batch: texts shape = {texts.shape}, labels shape = {labels.shape}")

                optimizer.zero_grad()
                outputs = model(texts)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * texts.size(0)
                predicted = torch.round(outputs.squeeze())
                correct_train += (predicted == labels).sum().item()
                total_train += texts.size(0)

            train_loss_list.append(train_loss / total_train)
            train_acc_list.append(correct_train / total_train)

            model.eval()
            correct_test = 0
            total_test = 0

            with torch.no_grad():
                for texts, labels in test_loader:
                    texts, labels = texts.to(device), labels.to(device)

                    print(f"Testing batch: texts shape = {texts.shape}, labels shape = {labels.shape}")

                    outputs = model(texts)
                    predicted = torch.round(outputs.squeeze())
                    correct_test += (predicted == labels).sum().item()
                    total_test += texts.size(0)

            test_acc_list.append(correct_test / total_test)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss_list[-1]:.4f}, Train Acc: {train_acc_list[-1]:.4f}, Test Acc: {test_acc_list[-1]:.4f}")

        # Plotting results
        plt.plot(train_loss_list, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.show()

        plt.plot(train_acc_list, label="Train Acc")
        plt.plot(test_acc_list, label="Test Acc")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Test Accuracy")
        plt.legend()
        plt.show()

        # Confusion Matrix
        model.eval()
        y_pred = []
        with torch.no_grad():
            for texts, labels in test_loader:
                texts = texts.to(device)
                outputs = model(texts)
                predicted = torch.round(outputs.squeeze()).tolist()
                y_pred.extend(predicted)

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
    else:
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                predicted = torch.round(outputs.squeeze())
                correct_test += (predicted == labels).sum().item()
                total_test += texts.size(0)

        test_acc = correct_test / total_test
        print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
