# src/gui.py
import tkinter as tk
from tkinter import ttk
import torch

def predict_winner(model, ufc_data, fighter_1, fighter_2):
    fighter_1_data = ufc_data[ufc_data['BlueFighter'] == fighter_1].iloc[0]
    fighter_2_data = ufc_data[ufc_data['RedFighter'] == fighter_2].iloc[0]
    
    # Prepare all 6 input features for the model
    X_fighter = torch.tensor([
        float(fighter_1_data['BlueLosses']),
        float(fighter_2_data['RedLosses']),
        float(fighter_1_data['BlueWinLossPercentage']),
        float(fighter_2_data['RedWinLossPercentage']),
        float(fighter_1_data['BlueCurrentWinStreak']),
        float(fighter_2_data['RedCurrentWinStreak']),
    ], dtype=torch.float32).unsqueeze(0)

    # Make the prediction
    model.eval()
    with torch.no_grad():
        output = model(X_fighter)
        win_probability = output.item() * 100  

    return win_probability


def create_gui(model, ufc_data):
    def on_predict():
        fighter_1 = fighter_1_var.get()
        fighter_2 = fighter_2_var.get()
        if fighter_1 != fighter_2:
            win_prob = predict_winner(model, ufc_data, fighter_1, fighter_2)
            result_label.config(text=f"Chance of {fighter_1} winning: {win_prob:.2f}%")
        else:
            result_label.config(text="Please select two different fighters.")

    root = tk.Tk()
    root.title("UFC Fight Predictor")

    fighter_1_label = tk.Label(root, text="Select Fighter 1:")
    fighter_1_label.pack()
    fighter_1_var = tk.StringVar()
    fighter_1_dropdown = ttk.Combobox(root, textvariable=fighter_1_var, values=sorted(ufc_data['BlueFighter'].unique()))
    fighter_1_dropdown.pack()

    fighter_2_label = tk.Label(root, text="Select Fighter 2:")
    fighter_2_label.pack()
    fighter_2_var = tk.StringVar()
    fighter_2_dropdown = ttk.Combobox(root, textvariable=fighter_2_var, values=sorted(ufc_data['RedFighter'].unique()))
    fighter_2_dropdown.pack()

    predict_button = tk.Button(root, text="Predict Winner", command=on_predict)
    predict_button.pack()

    # Label to show the result
    result_label = tk.Label(root, text="")
    result_label.pack()

    root.mainloop()
