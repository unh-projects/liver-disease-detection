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
                         'Alanine_Aminotransferase',
                         'Aspartate_Aminotransferase', 
                         'Total_Proteins', 
                         'Albumin',
                         'Albumin_and_Globulin_Ratio']
        
        self.target_col = 'Diagnosis'
        self.file_path = file_path
        
    
    def preprocess(self, df):
        # Rename columns for diagnosis classes
        df = df.rename(columns={'Dataset': 'Diagnosis'})
        df['Diagnosis'] = df['Diagnosis'].apply(lambda x:1 if x==1 else 0)

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
        # scaler = StandardScaler().fit(X)
        # X_scaled = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
        
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
        age = col1.number_input("Age", 
                            min_value=None,
                            value=72,
                            help="In the United States, the average age at onset of liver cancer is 63 years.")
        gender = col2.selectbox('Gender',
                            ('Male', 'Female'),
                            help="Men are more likely to develop liver cancer than women, by a ratio of 2 to 1.")
        total_bilirubin = col1.number_input("Total_Bilirubin (mg/dL)", 
                                            min_value=None,
                                            value=0.7, 
                                            help="It is normal to have some bilirubin in the blood. A normal level is: 0.1 to 1.2 mg/dL (1.71 to 20.5 µmol/L)")
        direct_bilirubin = col2.number_input("Direct_Bilirubin (mg/dL)", 
                                            min_value=None,
                                            value=0.1, 
                                            help="Normal level for Direct (also called conjugated) bilirubin is less than 0.3 mg/dL.")
        alkaline_phosphotase = col1.number_input("Alkaline_Phosphotase (IU/L)", 
                                            min_value=None,
                                            value=182,
                                            help="The normal range is 44 to 147 international units per liter (IU/L).")
        alanine_aminotransferase = col2.number_input("Alanine_Aminotransferase (U/L)", 
                                                    min_value=None,
                                                    value=24,
                                                    help="The normal range is 4 to 36 U/L.")
        aspartate_aminotransferase = col1.number_input("Aspartate_Aminotransferase (U/L)", 
                                                    min_value=None,
                                                    value=19,
                                                    help="The normal range is 8 to 33 U/L.")
        total_proteins = col2.number_input("Total_Proteins (g/dL)", 
                                        min_value=None,
                                        value=8.9,
                                        help="The normal range is 6.0 to 8.3 grams per deciliter (g/dL) or 60 to 83 g/L.")
        albumin = col1.number_input("Albumin (G/dL)", 
                                min_value=None,
                                value=4.9,
                                help="The normal range is 3.4 to 5.4 g/dL (34 to 54 g/L).")
        albumin_and_globulin_ratio = col2.number_input("Albumin_and_Globulin_Ratio", 
                                                    min_value=None,
                                                    value=1.20,
                                                    help="The normal range for albumin/globulin ratio is over 1 , usually around 1 to 2.")
        gender = 0 if gender == "Male" else 1

        return {'Age': age, 
                'Gender': gender, 
                'Total_Bilirubin': total_bilirubin, 
                'Direct_Bilirubin': direct_bilirubin, 
                'Alkaline_Phosphotase': alkaline_phosphotase, 
                'Alanine_Aminotransferase': alanine_aminotransferase, 
                'Aspartate_Aminotransferase': aspartate_aminotransferase, 
                'Total_Proteins': total_proteins, 
                'Albumin': albumin, 
                'Albumin_and_Globulin_Ratio': albumin_and_globulin_ratio}
            
        
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
                </style><div class="footer"><p>Developed by <a style='display: block; text-align: center;' href="#" target="_blank">Team Phoenix for DSCI-6002 Final Project</a></p></div>
                """

        return footer
        