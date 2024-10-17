# src/evaluate.py
import torch

def evaluate_model(model, X_test, y_test):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        model.eval()
        y_pred = model(X_test_tensor)
        y_pred = y_pred.round()

        # Calculate accuracy
        accuracy = (y_pred.eq(y_test_tensor).sum() / float(y_test_tensor.shape[0])).item()
        print(f'Accuracy: {accuracy * 100:.2f}%')
