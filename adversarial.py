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

class EspionAI(nn.Module):
    """modified model to allow for adversarial attacks"""
    def __init__(self, input_size, hidden_size, embedding_dim):
        super(EspionAI, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)  # Embedding layer
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1) 
        self.relu = nn.ReLU()

    def forward(self, x):
        # x is a tensor of indices (long)
        x = self.embedding(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.mean(dim=1)
        return x
    
    # New method to forward from embeddings
    def forward_from_embeddings(self, emb):
        # emb is the embeddings tensor
        x = self.fc1(emb)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.mean(dim=1)
        return x

def fgsm_attack_targeted(model, loss_fn, x, y_target, epsilon):
    """an adversarial attack on the EspionAI model using fgsm, returns pertubred embeddings"""
    x = x.to(device)
    y_target = y_target.to(device)
    
    # Get the embeddings from the model
    emb = model.embedding(x)
    
    # Detach and enable gradients on embeddings
    emb = emb.detach().clone().requires_grad_(True)
    
    # Forward pass using the embeddings
    logits = model.forward_from_embeddings(emb)
    
    # Compute the loss with respect to the target label
    y_target = y_target.float()
    loss = loss_fn(logits.view(-1), y_target)
    
    # Backward pass to compute gradients w.r.t. embeddings
    model.zero_grad()
    loss.backward()
    
    # Collect the gradients of the embeddings
    grad = emb.grad.data
    
    # Create adversarial example by perturbing embeddings in the negative gradient direction
    perturbed_emb = emb - epsilon * grad.sign()  
    
    # Return the perturbed embeddings
    return perturbed_emb.detach()

def preprocess_input(input_text, tokenizer, vocab, max_length):
    """Preprocess text to input"""
    # Tokenize the input text
    tokenized_input = tokenizer(input_text)
    # Convert tokens to corresponding indices
    numerical_input = [vocab[token] for token in tokenized_input]
    # Pad or truncate the sequence
    padded_input = torch.tensor(numerical_input, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    padded_input = padded_input[:, :max_length]  # Truncate if necessary
    return padded_input

def inference_from_embeddings(embeddings, model, device):
    """function for performing inference from embeddings using model, returns predictions"""
    embeddings = embeddings.to(device)
    model.eval()
    with torch.no_grad():
        logits = model.forward_from_embeddings(embeddings)
        probabilities = torch.sigmoid(logits)
        prediction = (probabilities > 0.5).cpu().numpy().astype(int)
    return prediction

def inference(input_text, model, tokenizer, vocab, max_length, device):
    """inference function, returns predictions"""
    # Preprocess the input text
    processed_input = preprocess_input(input_text, tokenizer, vocab, max_length)
    # Move to the correct device
    processed_input = processed_input.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        logits = model(processed_input)
        probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
        prediction = (probabilities > 0.5).cpu().numpy().astype(int)  # Threshold to get binary prediction
    
    return prediction

def perturb_input(input_text, model, tokenizer, vocab, max_length, epsilon, device):
    """returns perturbed embeddings from input of text"""
    # Preprocess the input text
    processed_input = preprocess_input(input_text, tokenizer, vocab, max_length)
    processed_input = processed_input.to(device)
    
    # Specify the target label (e.g., 0)
    target_label = torch.tensor([0], dtype=torch.float32).to(device)
    
    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Perform targeted FGSM attack to generate perturbed embeddings
    perturbed_embeddings = fgsm_attack_targeted(model, loss_fn, processed_input, target_label, epsilon)
    
    return perturbed_embeddings


def load_model(path):
    """load my pre trained model, send to cpu because limitations on mps support"""
    model = torch.load(path, map_location='cpu')

    # Move the model to the device (CPU or GPU)
    device = torch.device("cpu")

    return model, device

model, device = load_model('espionai_model.pth')
model.to(device)

# Preprocessing steps (same as before)
def csv_to_text(filename):
    """csv to texts"""
    intel_data = pd.read_csv(filename)

    intel_data.drop(labels = ['True', 'Doc Title'], axis = 1, inplace = True)

    intel_data['Threat level'] = np.where(intel_data['Threat level'] == 'High', 1, 0)

    texts = intel_data['text']

    return texts

texts = csv_to_text("intelligence_data.csv")

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

    return vocab_obj, tokenizer, max_length

vocab_obj, tokenizer, max_length = tokenise(texts)




# Example usage
input_text = texts[0]  
prediction = inference(input_text, model, tokenizer, vocab_obj, max_length, device)
print(f"Original Predicted Threat level: {'High' if prediction == 1 else 'Low'}")

# Perturb the input toward the target label 0
epsilon = 0.03  # Define the magnitude of the perturbation
perturbed_embeddings = perturb_input(input_text, model, tokenizer, vocab_obj, max_length, epsilon, device)

# Now you can run inference on the perturbed embeddings
perturbed_prediction = inference_from_embeddings(perturbed_embeddings, model, device)
print(f"Predicted Threat level for perturbed input: {'High' if perturbed_prediction == 1 else 'Low'}")