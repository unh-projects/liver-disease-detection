import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class Util:
    def __init__(self, file_path = 'data/indian_liver_patient.csv'):
        self.features = ['Age', 
                         'Gender', 
                         'Total_Bilirubin', 
                         'Direct_Bilirubin',
                         'Alkaline_Phosphotase', 
                         'Alamine_Aminotransferase',
                         'Aspartate_Aminotransferase', 
                         'Total_Protiens', 
                         'Albumin',
                         'Albumin_and_Globulin_Ratio']
        
        self.target_col = 'Diagnosis'
        self.file_path = file_path
        
    
    def preprocess(self, df):
        # Rename columns for diagnosis classes
        df = df.rename(columns={'Dataset': 'Diagnosis'})
        
        # Fill null values
        mean_ratio = df['Albumin_and_Globulin_Ratio'].mean()
        df = df.fillna(mean_ratio)

        # Convert categorical to numerical column
        df['Gender'] = df['Gender'].apply(lambda x:1 if x=='Male' else 0)
        
        return df
        
    def get_data(self):
        df = pd.read_csv(self.file_path)
        
        # preprocess data
        df = self.preprocess(df)
        
        X = df[self.features]
        y = df[self.target_col]

        # Scaling the feature columns
        # scaler = StandardScaler().fit(X)
        # X_scaled = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
        
        return X_train, X_test, y_train, y_test
        
    def build_model(self):
        
        X_train, X_test, y_train, y_test = self.get_data()
        model = LogisticRegression()
        
        print("Fitting the model")
        model.fit(X_train, y_train)
                
        return model
    
    def compute_accuracy(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)*100
    
    def predict(model, X):
        prediction = model.predict(X)
        return prediction
        
        

        