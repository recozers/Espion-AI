import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu") #Change to cuda if gpu
print(f"Using device: {device}")

def csv_to_text(filename):
    """csv to texts"""
    intel_data = pd.read_csv(filename)

    intel_data.drop(labels = ['True', 'Doc Title'], axis = 1, inplace = True)

    intel_data['Threat level'] = np.where(intel_data['Threat level'] == 'High', 1, 0)

    texts = intel_data['text']

    return texts, intel_data

texts, intel_data = csv_to_text("intelligence_data.csv")

def tokenise(texts):
    """tokenises the texts and returns the vocab, tokeniser and max_length"""
    tokenizer = get_tokenizer("basic_english")
    tokenized_texts = [tokenizer(text) for text in texts]

    counter = Counter(token for tokens in tokenized_texts for token in tokens)
    vocab_obj = vocab(counter, specials=['<pad>'])

    numerical_sequences = [torch.tensor([vocab_obj[token] for token in tokens]) for tokens in tokenized_texts]

    sequence_lengths = [len(seq) for seq in numerical_sequences]
    max_length = int(np.percentile(sequence_lengths, 50))  # Standard length for input to network

    padded_sequences = pad_sequence(numerical_sequences, batch_first=True, padding_value=vocab_obj['<pad>'])
    padded_and_truncated_sequences = padded_sequences[:, :max_length]

    return vocab_obj, tokenizer, max_length, padded_and_truncated_sequences

vocab_obj, tokenizer, max_length, padded_and_truncated_sequences = tokenise(texts)

X = padded_and_truncated_sequences
y = intel_data['Threat level']

## train model

class EspionAI(nn.Module):
    """DNN with embedding layer, hidden layer and output layer, returns raw logits"""
    def __init__(self, input_size, hidden_size, embedding_dim):
        super(EspionAI, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)  # Embedding layer
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1) 
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Define the forward pass
        x = self.embedding(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.mean(dim=1)
        #x = self.sigmoid(x)
        return x

def train_model(embedding_dims, X, y, test_size, learning_rate, epochs):
    """training loop for Espion-AI with variable inputs, returns the trained model and the test data"""

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size, random_state=0)

    #training loop 
    torch.mps.empty_cache()

    X_train = torch.tensor(X_train, dtype=torch.long).to(device)
    y_train = y_train.values
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    tot_batch = len(dataloader)

    model = EspionAI(input_size=max_length, hidden_size=max_length, embedding_dim=embedding_dims)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            outputs = outputs.view(-1)

            loss = criterion(outputs, batch_y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
        tot_loss = epoch_loss/tot_batch
        print("Epoch loss :", tot_loss)

    return model, X_test, y_test

model, X_test, y_test = train_model(50,X,y,0.2,0.001,50) # I enocurage you to play around with these

def evaluate(X_test, y_test, model):
    """evaluate the models performance"""

    X_test = torch.tensor(X_test, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        y_hat = model(X_test)  # Raw logits
        y_hat = torch.sigmoid(y_hat)  # Convert logits to probabilities
        y_pred = (y_hat > 0.5).cpu().numpy().astype(int)  # Threshold probabilities

    # Calculate accuracy
    acc = classification_report(y_test, y_pred)  # Ensure y_test is on CPU
    print(acc)

evaluate(X_test, y_test, model)

def save_model(model):
    """saves the model to be loaded elsewhere, use this if you want to overwrite the model in the repo"""
    torch.save(model, "espionai_model.pth")