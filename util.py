import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import streamlit as st
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
                         'Total_Proteins', 
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
    
    @st.cache
    def get_data(self):
        df = pd.read_csv(self.file_path)
        
        # preprocess data
        df = self.preprocess(df)
        
        X = df[self.features]
        y = df[self.target_col]

        # Scaling the feature columns
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=21)
        
        return X_train, X_test, y_train, y_test

    @st.cache  
    def build_model(self, X, y):
        model = LogisticRegression()
        
        print("Fitting the model")
        model.fit(X, y)
                
        return model
    
    def compute_accuracy(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)*100
    
    def predict(model, X):
        prediction = model.predict(X)
        return prediction

    def input_data_fields(self):

        col1, col2 = st.columns(2)
        age = col1.text_input("Age", 72)
        gender = col2.selectbox('Gender',('Male', 'Female'))
        total_bilirubin = col1.text_input("Total_Bilirubin", 0.7)
        direct_bilirubin = col2.text_input("Direct_Bilirubin", 0.1)
        alkaline_phosphotase = col1.text_input("Alkaline_Phosphotase", "182")
        alamine_aminotransferase = col2.text_input("Alamine_Aminotransferase", 24)
        aspartate_aminotransferase = col1.text_input("Aspartate_Aminotransferase", 19)
        total_proteins = col2.text_input("Total_Proteins", 8.9)
        albumin = col1.text_input("Albumin", 4.9)
        albumin_and_globulin_ratio = col2.text_input("Albumin_and_Globulin_Ratio", 1.20)

        return {'age': age, 
                'gender': gender, 
                'total_bilirubin': total_bilirubin, 
                'direct_bilirubin': direct_bilirubin, 
                'alkaline_phosphotase': alkaline_phosphotase, 
                'alamine_aminotransferase': alamine_aminotransferase, 
                'aspartate_aminotransferase': aspartate_aminotransferase, 
                'total_proteins': total_proteins, 
                'albumin': albumin, 
                'albumin_and_globulin_ratio': albumin_and_globulin_ratio}
            
        
    def page_footer(self):
        footer="""<style>
                a:link , a:visited{
                color: blue;
                background-color: transparent;
                text-decoration: underline;
                }

                a:hover,  a:active {
                color: red;
                background-color: transparent;
                text-decoration: underline;
                }

                .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: white;
                color: black;
                text-align: center;
                }
                </style><div class="footer"><p>Developed with ❤ by <a style='display: block; text-align: center;' href="#" target="_blank">Team Phoenix for DSCI-6002 Final Project</a></p></div>
                """

        return footer
        