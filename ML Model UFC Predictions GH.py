import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import ttk

file_path = r"C:\Users\Gebruiker\OneDrive\Documents\Python UFC Machine Learning\ufc new dataset\ufc-master.csv"
ufc_data = pd.read_csv(file_path)
ufc_data = ufc_data.sort_values(by='Date') 

def calculate_win_loss_percentage(data):
    data['BlueWinLossPercentage'] = 0.0
    data['RedWinLossPercentage'] = 0.0
    cumulative_stats = {}
    
    for idx, row in data.iterrows():
        blue_fighter = row['BlueFighter']  
        red_fighter = row['RedFighter']    
        
        # Initialize fighters if not already in stats
        if blue_fighter not in cumulative_stats:
            cumulative_stats[blue_fighter] = {'wins': 0, 'losses': 0}
        if red_fighter not in cumulative_stats:
            cumulative_stats[red_fighter] = {'wins': 0, 'losses': 0}
        
        # Calculate win/loss percentage before the current fight
        blue_wins = cumulative_stats[blue_fighter]['wins']
        blue_losses = cumulative_stats[blue_fighter]['losses']
        red_wins = cumulative_stats[red_fighter]['wins']
        red_losses = cumulative_stats[red_fighter]['losses']
        
        if blue_wins + blue_losses > 0:
            blue_win_loss_percentage = blue_wins / (blue_wins + blue_losses)
        else:
            blue_win_loss_percentage = 0.0  

        if red_wins + red_losses > 0:
            red_win_loss_percentage = red_wins / (red_wins + red_losses)
        else:
            red_win_loss_percentage = 0.0  

        # Update the dataframe with calculated percentages
        data.at[idx, 'BlueWinLossPercentage'] = blue_win_loss_percentage
        data.at[idx, 'RedWinLossPercentage'] = red_win_loss_percentage
        
        # Update win/loss record based on the current fight outcome
        if row['Winner'] == 'Blue':  
            cumulative_stats[blue_fighter]['wins'] += 1
            cumulative_stats[red_fighter]['losses'] += 1
        else:
            cumulative_stats[blue_fighter]['losses'] += 1
            cumulative_stats[red_fighter]['wins'] += 1

    return data

ufc_data = calculate_win_loss_percentage(ufc_data)

# Keep only necessary columns
ufc_data = ufc_data[['Winner', 'BlueLosses', 'RedLosses', 'RedWinLossPercentage', 
                     'BlueCurrentWinStreak', 'RedCurrentWinStreak', 'RedLongestWinStreak',
                     'BlueFighter', 'RedFighter']]

# Drop rows with missing values in any of the selected columns
ufc_data.dropna(subset=['Winner', 'BlueLosses', 'RedLosses', 'RedWinLossPercentage', 
                        'BlueCurrentWinStreak', 'RedCurrentWinStreak', 'RedLongestWinStreak', 
                        'BlueFighter', 'RedFighter'], inplace=True)

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode the 'Winner', 'BlueFighter', and 'RedFighter' columns
ufc_data['Winner'] = label_encoder.fit_transform(ufc_data['Winner'])  # Blue=1, Red=0
ufc_data['BlueFighter'] = label_encoder.fit_transform(ufc_data['BlueFighter'])
ufc_data['RedFighter'] = label_encoder.fit_transform(ufc_data['RedFighter'])

# Define features (X) and target (y) after encoding
X = ufc_data[['BlueLosses', 'RedLosses', 'RedWinLossPercentage', 'BlueCurrentWinStreak', 'RedCurrentWinStreak', 'RedLongestWinStreak']].values
y = ufc_data['Winner'].values



# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Instantiate the model and optimizer
model = SimpleNN()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Training loop
num_epochs = 80
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.round() 

    # Calculate accuracy
    accuracy = (y_pred.eq(y_test).sum() / float(y_test.shape[0])).item()
    print(f'Accuracy: {accuracy * 100:.2f}%')

# GUI Code to predict fighter win probability
def predict_winner(fighter_1, fighter_2):
    fighter_1_data = ufc_data[ufc_data['BlueFighter'] == fighter_1].iloc[0]
    fighter_2_data = ufc_data[ufc_data['RedFighter'] == fighter_2].iloc[0]
    
    # Prepare the input data for the model
    X_fighter = torch.tensor([
        float(fighter_1_data['BlueLosses']),
        float(fighter_2_data['RedLosses']),
    ], dtype=torch.float32).unsqueeze(0)

    # Make the prediction
    model.eval()
    with torch.no_grad():
        output = model(X_fighter)
        win_probability = output.item() * 100  

    return win_probability

def create_gui():
    def on_predict():
        fighter_1 = fighter_1_var.get()
        fighter_2 = fighter_2_var.get()
        if fighter_1 != fighter_2:
            win_prob = predict_winner(fighter_1, fighter_2)
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

# Uncomment to run the GUI
create_gui()
