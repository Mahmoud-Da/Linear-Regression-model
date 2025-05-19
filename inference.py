import torch

import config
from model import LinearRegressionModel
from utils import plot_predictions
from data_setup import create_dataset


def main():
    print(f"Using device: {config.DEVICE}")

    # Set the random seed for reproducibility of dataset generation
    # This ensures X_train, y_train, X_test, y_test are the same as in train.py
    # if create_dataset() is deterministic given the seed.
    torch.manual_seed(config.RANDOM_SEED)
    print(f"Set random seed to: {config.RANDOM_SEED}")

    # 1. Create an instance of the model
    # This needs to be the same architecture as the saved model
    loaded_model = LinearRegressionModel().to(config.DEVICE)

    # 2. Load the saved state_dict
    print(f"Loading model state from: {config.MODEL_SAVE_PATH}")
    try:
        # map_location ensures model loads correctly even if trained on different device
        loaded_model.load_state_dict(torch.load(
            f=config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {config.MODEL_SAVE_PATH}")
        print("Please run train.py first to train and save the model.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # 3. Set the model to evaluation mode
    loaded_model.eval()
    print("Model loaded successfully and set to evaluation mode.")

    # Check out our loaded model's parameters
    print("\nLoaded model parameters (state_dict):")
    print(loaded_model.state_dict())
    # print(list(loaded_model.parameters())) # For a more compact view of parameters

    # 4. Prepare data for inference (e.g., the test set or new unseen data)
    # We also get X_train and y_train here for plotting context.
    # The model was trained on X_train, y_train. We're using X_test for predictions.
    X_train, y_train, X_test, y_test, _, _ = create_dataset()

    # Move data to target device
    X_train = X_train.to(config.DEVICE)
    # y_train is for plotting ground truth of training data
    y_train = y_train.to(config.DEVICE)
    X_test = X_test.to(config.DEVICE)
    # y_test is for plotting ground truth of test data
    y_test = y_test.to(config.DEVICE)

    print(f"\nShape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 5. Make predictions
    with torch.inference_mode():
        y_preds_loaded = loaded_model(X_test)

    print(f"\nPredictions on X_test[:10]:\n{y_preds_loaded[:10]}")
    print(f"Actual y_test[:10]:\n{y_test[:10]}")

    # 6. Plot predictions (optional, but good for visualization)
    # Now we pass X_train and y_train to plot_predictions
    print("\nPlotting predictions with training data context:")
    plot_predictions(train_data=X_train,
                     train_labels=y_train,  # Make sure your plot_predictions can handle/use this
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_preds_loaded,
                     device=config.DEVICE)
    print("\nInference complete. Plot displayed.")

    # Example of predicting on a single new data point
    # Ensure the new data point is on the correct device
    # new_single_data_point = torch.tensor([[0.95]], dtype=torch.float32).to(config.DEVICE)
    # with torch.inference_mode():
    #    single_prediction = loaded_model(new_single_data_point)
    # print(f"\nPrediction for input {new_single_data_point.item()}: {single_prediction.item()}")


if __name__ == "__main__":
    main()
