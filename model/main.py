# pip3 install pandas
# pip3 install scikit-learn
# pip3 install --upgrade pip

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

#   _____________________________________
#   |   Clean data

# Read the data 
def get_clean_data():
    data = pd.read_csv("data/data.csv") # Use the csv from data folder from this streamlit root folder so the path is simple
    
    # Drop unnecessary columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # From the diagnosis column, change the values M,B to 0,1 by mapping from a dictionary
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })

    return data

#   _____________________________________
#   |   Train the model using clean data

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Standardization features or StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and testing set 20% of dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print('Accuracy of the model: ', accuracy_score(y_test, y_pred)) # accuracy_score function takes 1st parameter as actual value, 2nd parameter is your prediction value
    print("Classification report: \n", classification_report(y_test, y_pred))
    
    return model, scaler

# Main function including all phases
def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    # Save the model and the scaler file as pkl files in model folder
    with open('model/model.pkl', 'wb') as File: # with binary = wb -- write on the file with binary
        pickle.dump(model, File) # dump the model inside the file

    with open('model/scaler.pkl', 'wb') as File:
        pickle.dump(scaler, File) # dump the model inside the file -- write on the file with binary


# Test to ensure the correct file is executed 
if __name__ == '__main__':
    main()