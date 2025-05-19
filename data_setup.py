import torch
import config


def create_dataset():
    """Creates the feature and target tensors and splits them."""
    X = torch.arange(config.START, config.END, config.STEP).unsqueeze(dim=1)
    y = config.WEIGHT * X + config.BIAS

    # Create a train/test split
    train_split_index = int(config.TRAIN_SPLIT_RATIO * len(X))
    X_train, y_train = X[:train_split_index], y[:train_split_index]
    X_test, y_test = X[train_split_index:], y[train_split_index:]

    print(f"Generated data: {len(X)} samples.")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    # X[:10], y[:10] # Original check, can be printed if needed
    # Return original X, y for full data plotting if needed
    return X_train, y_train, X_test, y_test, X, y
