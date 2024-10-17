# src/data_processing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

def preprocess_data(file_path):
    ufc_data = pd.read_csv(file_path)
    ufc_data = calculate_win_loss_percentage(ufc_data)

    # Ensure proper encoding of categorical columns
    label_encoder = LabelEncoder()
    ufc_data['Winner'] = label_encoder.fit_transform(ufc_data['Winner'])  # Encodes to 0, 1

    # Prepare feature matrix X and target variable y
    X = ufc_data[['BlueLosses', 'RedLosses', 'RedWinLossPercentage', 'BlueCurrentWinStreak', 'RedCurrentWinStreak', 'RedLongestWinStreak']].values
    y = ufc_data['Winner'].values.astype('float32')  # Ensure y is of float32 type

    return X, y, ufc_data
