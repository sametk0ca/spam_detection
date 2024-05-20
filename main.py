import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader

# CUDA ayarlaması
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri setinin bulunduğu klasör
DATA_DIR = "C:\\code\\ai\\spam\\enron"

# LSTM modelinin parametreleri
EMBEDDING_DIM = 30
HIDDEN_DIM = 128
OUTPUT_DIM = 1
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 16

def read_text_files(directory):
    texts = []

    # 'enron' klasöründeki alt klasörleri gez
    for enron_folder in os.listdir(directory):
        enron_path = os.path.join(directory, enron_folder)

        # Eğer bir klasörse devam et
        if os.path.isdir(enron_path):
            
            # Alt klasördeki 'ham' ve 'spam' klasörlerini gez
            for category in ['ham', 'spam']:
                category_path = os.path.join(enron_path, category)
                
                # Eğer 'ham' ya da 'spam' klasörü varsa, dosyaları oku
                if os.path.exists(category_path):
                    for filename in os.listdir(category_path):
                        filepath = os.path.join(category_path, filename)
                        with open(filepath, "r", encoding="latin-1") as file:
                            text = file.read()
                            texts.append(text)
    return texts

def preprocess_text(texts):
    # Küçük harfe dönüştürme
    texts = [sentence.lower() for sentence in texts]
    # Noktalama işaretlerini ve özel karakterleri kaldırma
    texts = [' '.join(re.sub(r'[^\w\s]', '', sentence).split()) for sentence in texts]
    # Durak kelimelerini kaldırma
    stop_words = set(stopwords.words('english'))
    texts = [' '.join([word for word in word_tokenize(sentence) if word not in stop_words]) for sentence in texts]
    # URL'leri kaldırma
    texts = [re.sub(r'http\S+|www\S+|https\S+', '', sentence) for sentence in texts]
    # HTML etiketlerini kaldırma
    texts = [re.sub(r'<.*?>', '', sentence) for sentence in texts]
    # Metni parçalara ayırma (tokenization)
    texts = [word_tokenize(sentence) for sentence in texts]
    return texts

def text_to_sequence(texts):
    vocab = set()
    for text in texts:
        vocab.update(text)
    word_to_index = {word: idx + 1 for idx, word in enumerate(vocab)}
    sequences = [[word_to_index[word] for word in text] for text in texts]
    return sequences, vocab

def load_data(data_dir):
    ham_texts = []
    spam_texts = []
    labels = []

    # Veriyi oku
    for enron_folder in os.listdir(DATA_DIR):
        enron_path = os.path.join(data_dir, enron_folder)
        if os.path.isdir(enron_path):
            
            for category in os.listdir(enron_path):
                if category == "ham":
                    ham_texts.extend(read_text_files(DATA_DIR))
                    labels.extend([0] * len(read_text_files(DATA_DIR)))  # Doğru sayıda etiket ekle
                elif category == "spam":
                    spam_texts.extend(read_text_files(DATA_DIR))
                    labels.extend([1] * len(read_text_files(DATA_DIR)))  # Doğru sayıda etiket ekle

    return ham_texts, spam_texts, labels

# Veri setini tensörlere dönüştürmek için özel bir Dataset sınıfı oluşturma
class TextDataset(Dataset):
    def __init__(self, texts, labels, max_length, transform=None):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.transform:
            text = self.transform(text)
        # Padding with zeros to make each text tensor have the same length
        padded_text = self.pad_sequence(text, self.max_length)
        return padded_text, label

    def pad_sequence(self, sequence, max_length):
        padded_sequence = np.zeros(max_length, dtype=int)
        padded_sequence[:len(sequence)] = sequence
        return padded_sequence


# LSTM Modeli
class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, vocab_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden_state = lstm_out[:, -1, :]  # Son zaman adımının çıktısını al
        output = self.fc(last_hidden_state)  # Çıktıyı doğrudan çıktı boyutuna dönüştür
        return self.sigmoid(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_model", action="store_true", help="Train the model if this flag is provided.")
    args = parser.parse_args()
    
    # Veriyi yükle
    ham_texts, spam_texts, labels = load_data(DATA_DIR)
    
    # Metinleri temizle ve tokenize et
    ham_texts_cleaned = preprocess_text(ham_texts)
    spam_texts_cleaned = preprocess_text(spam_texts)
    
    # Ham ve spam verilerini birleştir
    all_texts = ham_texts_cleaned + spam_texts_cleaned
    all_labels = np.array(labels)
    
    # Metinleri sayı dizilerine dönüştür
    sequences, vocab = text_to_sequence(all_texts)
    
    # Veriyi train ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(sequences, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
    
    # Dataset ve DataLoader kullanarak veriyi yükleyin
    padding_value = 20
    max_length = max(len(sequence) for sequence in sequences) + padding_value

    train_dataset = TextDataset(X_train, y_train, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset = TextDataset(X_test, y_test, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # LSTM modelini oluştur
    model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, len(vocab)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if args.train_model:
        # Eğitim ve test performanslarını kaydetmek için listeler
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []
        
        # Eğitim
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for texts, labels in train_loader:
                texts = torch.tensor(texts, dtype=torch.long).to(device)
                labels = torch.tensor(labels, dtype=torch.float).to(device)
                
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
            
            # Test
            model.eval()
            correct_test = 0
            total_test = 0
            
            with torch.no_grad():
                for texts, labels in test_loader:
                    texts = torch.tensor(texts, dtype=torch.long).to(device)
                    labels = torch.tensor(labels, dtype=torch.float).to(device)
                    
                    outputs = model(texts)
                    predicted = torch.round(outputs.squeeze())
                    correct_test += (predicted == labels).sum().item()
                    total_test += texts.size(0)
            
            test_acc_list.append(correct_test / total_test)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss_list[-1]:.4f}, Train Acc: {train_acc_list[-1]:.4f}, Test Acc: {test_acc_list[-1]:.4f}")
        
        # Loss grafiği
        plt.plot(train_loss_list, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.show()
        
        # Acc grafiği
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
                texts = torch.tensor(texts, dtype=torch.long).to(device)
                
                outputs = model(texts)
                predicted = torch.round(outputs.squeeze()).tolist()
                y_pred.extend(predicted)
        
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
    else:
        # Modeli eğitme argümanı olmadan test aşamasına geç
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for texts, labels in test_loader:
                texts = torch.tensor(texts, dtype=torch.long).to(device)
                labels = torch.tensor(labels, dtype=torch.float).to(device)
                
                outputs = model(texts)
                predicted = torch.round(outputs.squeeze())
                correct_test += (predicted == labels).sum().item()
                total_test += texts.size(0)
        
        test_acc = correct_test / total_test
        print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
