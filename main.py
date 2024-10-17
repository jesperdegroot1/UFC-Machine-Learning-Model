# main.py
from src.data_processing import preprocess_data
from src.model import SimpleNN
from src.train import train_model
from src.evaluate import evaluate_model
from src.gui import create_gui

def main():
    # Step 1: Load and preprocess the data
    file_path = r"C:\Users\Gebruiker\OneDrive\Documents\Python UFC Machine Learning\Python step by step\GH Upload\data\ufc-master.csv"
    X, y, ufc_data = preprocess_data(file_path)  # Unpack three values: X, y, and ufc_data

    # Step 2: Initialize the model
    model = SimpleNN()

    # Step 3: Train the model
    train_model(model, X, y)

    # Step 4: Evaluate the model
    evaluate_model(model, X, y)

    # Step 5: Run the GUI for predictions
    create_gui(model, ufc_data)

if __name__ == "__main__":
    main()