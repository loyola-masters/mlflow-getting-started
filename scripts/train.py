import os
import sys
import torch
from torch import nn

import mlflow
import mlflow.pytorch
from datetime import datetime

from model import build_model, device
from dataset import build_dataloaders

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
mlflow.set_tracking_uri(tracking_uri)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    test_acc = correct * 100.0
    print(f"Test Error: \n Accuracy: {(test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_acc, test_loss


def run_training(epochs=3, learning_rate=1e-2, batch_size=64):
    print(f"Entrenando con {epochs} epochs, LR={learning_rate}, Batch size={batch_size}")
    train_dataloader, test_dataloader = build_dataloaders(batch_size)
    model, signature = build_model()
    
    # Print signature
    print("Signature:", signature)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    best_acc = 0.0

    # Generate experiment run with timestamp
    experiment_name = "MNIST_experiment"
    mlflow.set_experiment(experiment_name)
    
    # Generate run name with timestamp
    run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("device", device)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test_acc, test_loss = test(test_dataloader, model, loss_fn)

            mlflow.log_metric("test_acc", test_acc, step=t)
            mlflow.log_metric("test_loss", test_loss, step=t)

            if test_acc > best_acc:
                best_acc = test_acc
                mlflow.log_metric("best_acc", best_acc, step=t)
                mlflow.pytorch.log_model(model.cpu(), "model", signature=signature, code_paths=["scripts/model.py"])
                model.to(device)

        # Save model to file
        model_path = "model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Modelo guardado en {model_path}")

    print("Done!")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo con PyTorch y MLFlow")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas de entrenamiento")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Tasa de aprendizaje")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamaño del batch")

    args = parser.parse_args()
    run_training(args.epochs, args.learning_rate, args.batch_size)
