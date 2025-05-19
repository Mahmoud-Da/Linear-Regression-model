# Linear Regression with PyTorch

This project demonstrates a simple machine learning workflow using PyTorch, including training and inference with a `LinearRegressionModel`.

## 📁 Project Structure

Ensure the following files are saved in the **same directory** (e.g., `your_project_folder`):

- `config.py`
- `data_setup.py`
- `model.py`
- `utils.py`
- `train.py`
- `inference.py`

---

## 🚀 How to Run

### 1. Navigate to Your Project Directory

Open your terminal or command prompt and navigate to the folder where the files are located:

```bash
cd your_project_folder
```

---

### 2. Train the Model

Run the following command to train the model:

```bash
python3 train.py
```

#### This will:

- Create the dataset.
- Train the `LinearRegressionModel`.
- Plot the loss curves and final predictions.
- Create a `models/` directory (if it doesn't already exist).
- Save the trained model’s `state_dict` to:
  `models/01_pytorch_workflow_model_0.pth`

---

### 3. Run Inference

After training, you can run inference using:

```bash
python3 inference.py
```

#### This will:

- Load the `LinearRegressionModel` architecture.
- Load the saved `state_dict` from the file.
- Make predictions on the test data (or new data if modified).
- Plot the predictions.

#### Result
- ![Screenshot 2025-05-19 at 20 48 22](https://github.com/user-attachments/assets/354a13db-e51f-4258-8673-e0f9954e746c)

---
