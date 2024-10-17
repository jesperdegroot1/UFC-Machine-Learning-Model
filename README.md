# UFC Fight Prediction Model
This project is a machine learning model that predicts the winner of UFC fights using fighter statistics. The model is built with PyTorch and trained using historical fight data. It also includes a graphical user interface (GUI) that allows users to select two fighters and predict the probability of a particular fighter winning.

Table of Contents
Introduction
Features
Dependencies
Installation
Usage
Model Details
GUI for Predictions
License
Introduction
This project is designed to predict the winner of UFC fights based on a variety of fighter statistics, such as win/loss percentages, win streaks, and losses. The model uses a simple feedforward neural network to make predictions based on this data. The application also includes a GUI that enables users to select two fighters and get a prediction on which one is more likely to win.

Features
A machine learning model using a neural network to predict UFC fight outcomes.
The model is trained on a dataset of historical UFC fights, using fighter statistics such as win percentages, losses, and win streaks.
A graphical user interface (GUI) that allows users to select fighters and see win probabilities.
Automatic encoding of fighter names and categorical variables.
Dependencies
The following libraries are required to run this project:

Python 3.x
Pandas
PyTorch
scikit-learn
Tkinter (for GUI)
torch.nn
torch.optim
Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/your-username/ufc-fight-prediction.git
cd ufc-fight-prediction
Install Required Packages:

Install the dependencies using pip:

bash
Copy code
pip install pandas torch scikit-learn tkinter
Dataset:

Make sure you have the ufc-master.csv dataset in the appropriate directory, or update the file_path variable in the script to point to the correct location of your dataset.

Usage
Run the Script:

To run the machine learning model and GUI for UFC fight predictions, simply execute the script:

bash
Copy code
python ufc_fight_prediction.py
Using the GUI:

Select two fighters from the dropdown menus.
Click "Predict Winner" to see the probability of the selected fighter winning the fight.
Training the Model:

The neural network model is trained using PyTorch with a simple feedforward network architecture. The dataset is split into training and testing sets, and the model is trained for 80 epochs.

Evaluate Model:

The model is evaluated based on accuracy on the test set after training.

Model Details
The machine learning model is a feedforward neural network built with PyTorch. It uses fighter statistics such as:

BlueLosses: Number of losses for the blue fighter.
RedLosses: Number of losses for the red fighter.
RedWinLossPercentage: Win/loss percentage for the red fighter.
BlueCurrentWinStreak: Current win streak for the blue fighter.
RedCurrentWinStreak: Current win streak for the red fighter.
RedLongestWinStreak: Longest win streak for the red fighter.
The neural network has 4 fully connected layers:

Input Layer: 6 features (fighter stats).
Hidden Layers: 3 layers with 16 neurons each.
Output Layer: 1 output (win probability for the blue fighter).
The model uses binary cross-entropy loss (BCELoss) and the Adam optimizer for training.

GUI for Predictions
The project includes a GUI built with Tkinter. The user selects two fighters, and the model predicts the chance of the first fighter (Blue Fighter) winning. The prediction is based on the fighter statistics fed into the neural network.

How to use the GUI:
Select Fighter 1 and Fighter 2 from the dropdown menus.
Click the "Predict Winner" button.
The result will display the predicted winning probability for Fighter 1.
