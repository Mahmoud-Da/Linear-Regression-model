# train.py
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

import config
from data_setup import create_dataset
from model import LinearRegressionModel
from utils import plot_predictions


def main():
    print(f"Using device: {config.DEVICE}")
    # Create a random seed
    torch.manual_seed(config.RANDOM_SEED)

    # 1. Create data
    X_train, y_train, X_test, y_test, X_all, y_all = create_dataset()
    # Move data to target device
    X_train, y_train = X_train.to(config.DEVICE), y_train.to(config.DEVICE)
    X_test, y_test = X_test.to(config.DEVICE), y_test.to(config.DEVICE)
    X_all, y_all = X_all.to(config.DEVICE), y_all.to(config.DEVICE)

    # 2. Build model
    model_0 = LinearRegressionModel().to(config.DEVICE)
    print("\nInitial model parameters:")
    # Check out the parameters
    print(list(model_0.parameters()))
    # List named parameters
    print(model_0.state_dict())

    # Plot initial predictions (before training)
    # Make predictions with model
    # with torch.inference_mode():
    # y_preds_initial = model_0(X_test)
    # plot_predictions(train_data=X_train, train_labels=y_train,
    # test_data=X_test, test_labels=y_test,
    # predictions=y_preds_initial, device=config.DEVICE)
    # print(f"Initial y_preds[:5]: {y_preds_initial[:5]}") # From original code: y_preds

    # 3. Setup loss function and optimizer
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(
        params=model_0.parameters(), lr=config.LEARNING_RATE)

    # 4. Training loop
    # An epoch is one loop through the data... (this is a hyperparameter because we've set it ourselves)
    epochs = config.EPOCHS

    # Track different values
    epoch_count = []
    loss_values = []
    test_loss_values = []

    print("\nStarting training...")
    # Training
    for epoch in range(epochs):
        # Set the model to training mode
        model_0.train()  # train mode in PyTorch sets all parameters that require gradients to require gradients

        # 1. Forward pass
        y_pred = model_0(X_train)

        # 2. Calculate the loss
        loss = loss_fn(y_pred, y_train)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Perform back-propagation on the loss with respect to the parameters of the model
        loss.backward()

        # 5. Step the optimizer (perform gradient descent)
        optimizer.step()

        # Testing
        model_0.eval()  # turns off different settings in the model not needed for evaluation/testing
        with torch.inference_mode():  # turns off gradient tracking
            # 1. Do the forward pass
            test_pred = model_0(X_test)
            # 2. Calculate the loss
            test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            # .item() gets the scalar value from a tensor
            loss_values.append(loss.item())
            test_loss_values.append(test_loss.item())
            print(
                f"Epoch: {epoch} | Loss: {loss:.4f} | Test loss: {test_loss:.4f}")
            # Print out model state_dict()
            # print(model_0.state_dict()) # This can be verbose, uncomment if needed

    print("\nTraining finished.")
    print("Final model state_dict():")
    print(model_0.state_dict())

    # Plot the loss curves
    plt.figure(figsize=(10, 7))  # Create a new figure for loss curves
    # np.array(torch.tensor(loss_values).numpy()) is redundant if storing .item()
    plt.plot(epoch_count, np.array(loss_values), label="Train loss")
    plt.plot(epoch_count, np.array(test_loss_values), label="Test loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # Make predictions with the trained model for plotting
    model_0.eval()
    with torch.inference_mode():
        y_preds_new = model_0(X_test)

    print("\nPlotting predictions after training:")
    plot_predictions(train_data=X_train, train_labels=y_train,
                     test_data=X_test, test_labels=y_test,
                     predictions=y_preds_new, device=config.DEVICE)

    # 5. Saving the model
    # 1. Create models directory
    config.MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path (already done in config)

    # 3. Save the model state dict
    print(f"\nSaving model to: {config.MODEL_SAVE_PATH}")
    torch.save(obj=model_0.state_dict(), f=config.MODEL_SAVE_PATH)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
