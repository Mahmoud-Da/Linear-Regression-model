import matplotlib.pyplot as plt
import torch  # For .cpu().numpy() if tensors are on GPU
from datetime import datetime


def plot_predictions(train_data=None,
                     train_labels=None,
                     test_data=None,
                     test_labels=None,
                     predictions=None,
                     device="cpu"):  # Add device to handle potential tensor location
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Ensure data is on CPU and converted to NumPy for plotting
    def to_cpu_numpy(tensor):
        if tensor is not None:
            return tensor.cpu().numpy()
        return None

    train_data_np = to_cpu_numpy(train_data)
    train_labels_np = to_cpu_numpy(train_labels)
    test_data_np = to_cpu_numpy(test_data)
    test_labels_np = to_cpu_numpy(test_labels)
    predictions_np = to_cpu_numpy(predictions)

    # Plot training data in blue
    if train_data_np is not None and train_labels_np is not None:
        plt.scatter(train_data_np, train_labels_np,
                    c="b", s=4, label="Training data")

    # Plot test data in green
    if test_data_np is not None and test_labels_np is not None:
        plt.scatter(test_data_np, test_labels_np,
                    c="g", s=4, label="Testing data")

    # Are there predictions?
    if predictions_np is not None and test_data_np is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data_np, predictions_np,
                    c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

    # plt.show()  # Add this to display the plot when run as a script without docker

    # Get the current date and time
    now = datetime.now()

    # Format the date and time into a string suitable for a filename
    # Example format: YYYY-MM-DD_HH-MM-SS
    # You can customize this format as needed.
    # See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    plot_filename = f"result_{timestamp_str}.png"  # Or .pdf, .jpg, etc.
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    # close the plot figure to free up memory, especially in scripts
    plt.close()
