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

# read the data first
def get_clean_data():
    data = pd.read_csv("data/data.csv") # just need to run the csv file from data folder in this streamlit root folder so no need longer path
    
    # drop not-needed columns, cleaning etc
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    #from the diagnosis column, change value of M,B (ac tinh vs lanh tinh) to 0,1 by mapping from a dictionary
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })

    return data

#   _____________________________________
#   |   Train the model using clean data

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # standardization features or StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split into training and testing set 20% of dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test the model
    y_pred = model.predict(X_test)
    print('Accuracy of the model: ', accuracy_score(y_test, y_pred)) # accuracy_score function takes 1st parameter as actual value, 2nd parameter is your prediction value
    print("Classification report: \n", classification_report(y_test, y_pred))
    
    return model, scaler

# Main function include all steps 
def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    # save the model and the scaler file as pickle files in model folder as well
    with open('model/model.pkl', 'wb') as File: #with binary = wb -- we gonna write on the file with binary
        pickle.dump(model, File) # dump the model inside the file

    with open('model/scaler.pkl', 'wb') as File:
        pickle.dump(scaler, File) # dump the model inside the file -- we gonna write on the file with binary



if __name__ == '__main__':
    main()