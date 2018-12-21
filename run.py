from dataset import XORDataset
from lstm_xor import LSTMXOR
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_TAG = "[Training]"
VALIDATION_TAG = "[Validation]"
BATCH_SIZE = 64

def run_model(model, optimizer, loss_function, dataset, epoch, tag, is_evaluation_mode=False, shuffle=True, batch_size=BATCH_SIZE):
    if is_evaluation_mode:
        model.eval()
    else:
        model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    with torch.set_grad_enabled(not is_evaluation_mode):
        current_correct_predictions = 0.0
        current_loss = []

        print("{} - [Epoch {}] - START".format(tag, epoch))

        for inputs, labels in tqdm(dataloader, desc=tag):
            inputs = inputs.to(DEVICE).float()
            labels = labels.to(DEVICE).float()

            if not is_evaluation_mode:
                optimizer.zero_grad()

            predictions = model(inputs)

            loss = loss_function(predictions, labels)
            current_loss.append(loss.cpu().data.item())
            current_correct_predictions += calculate_correct_predictions(predictions.cpu().data, labels.cpu().data)

            if not is_evaluation_mode:
                loss.backward()
                optimizer.step()
            
        print("{} - [Epoch {}] Accuracy is {:.5f}\n".format(tag, epoch, current_correct_predictions / len(dataset)))
        print("{} - [Epoch {}] Loss is {:.5f}\n".format(tag, epoch, sum(current_loss) / len(current_loss)))
        print("{} - [Epoch {}] - END\n\n".format(tag, epoch))

def calculate_correct_predictions(predictions, labels):
    predictions = torch.round(torch.sigmoid(predictions))
    correct_predictions = torch.sum(predictions == labels).data.item()

    return correct_predictions

if __name__ == "__main__":
    # Open dataset from CSV file
    training_df = pd.read_csv("dataset/training.csv")
    validation_df = pd.read_csv("dataset/validation.csv")
    X_train, y_train = training_df["inputs"].tolist(), training_df["labels"].tolist()
    X_val, y_val = validation_df["inputs"].tolist(), validation_df["labels"].tolist()

    # Start creating the datasets
    training_ds = XORDataset(X_train, y_train)
    validation_ds = XORDataset(X_val, y_val)

    # Create model, optimizer, and loss function
    lstm_xor_model = LSTMXOR(1, 512).to(DEVICE)
    optimizer = optim.Adam(lstm_xor_model.parameters(), lr=1e-2)
    loss_function = nn.BCEWithLogitsLoss()

    for epoch in range(3):
        # Train the model
        run_model(lstm_xor_model, optimizer, loss_function, training_ds, epoch + 1, TRAINING_TAG)

        # Validate the model
        run_model(lstm_xor_model, optimizer, loss_function, validation_ds, epoch + 1, VALIDATION_TAG, is_evaluation_mode=True, shuffle=False)
