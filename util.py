import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time

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
        df.rename(columns={'Dataset': 'Diagnosis'}, inplace=True)
        df[self.target_col] = df[self.target_col].apply(lambda x:1 if x==1 else 0)

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
        return df

    @st.cache    
    def split_data(self, df):
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

    def input_data_fields(self, overwrite_vals=None):

        default_vals = {'Age': 17,
                        'Gender': 1,
                        'Total_Bilirubin': 0.9,
                        'Direct_Bilirubin': 0.3,
                        'Alkaline_Phosphotase': 202,
                        'Alanine_Aminotransferase': 22,
                        'Aspartate_Aminotransferase': 19,
                        'Total_Proteins': 7.4,
                        'Albumin': 4.1,
                        'Albumin_and_Globulin_Ratio': 1.2}

        col1, col2 = st.columns(2)
        age = col1.number_input("Age", 
                            min_value=None,
                            step=5,
                            value=default_vals['Age'],
                            help="In the United States, the average age at onset of liver cancer is 63 years.")

        gender = col2.selectbox('Gender',
                            ('Male', 'Female'),
                            index=default_vals['Gender'],
                            help="Men are more likely to develop liver cancer than women, by a ratio of 2 to 1.")

        total_bilirubin = col1.number_input("Total_Bilirubin (mg/dL)", 
                                            min_value=None,
                                            step=0.5,
                                            value=default_vals['Total_Bilirubin'], 
                                            help="It is normal to have some bilirubin in the blood. A normal level is: 0.1 to 1.2 mg/dL (1.71 to 20.5 µmol/L)")
        
        direct_bilirubin = col2.number_input("Direct_Bilirubin (mg/dL)", 
                                            min_value=None,
                                            step=0.5,
                                           value=default_vals['Direct_Bilirubin'], 
                                            help="Normal level for Direct (also called conjugated) bilirubin is less than 0.3 mg/dL.")
   
        alkaline_phosphotase = col1.number_input("Alkaline_Phosphotase (IU/L)", 
                                            min_value=None,
                                            step=10,
                                            value=default_vals['Alkaline_Phosphotase'],
                                            help="The normal range is 44 to 147 international units per liter (IU/L).")
        
        alanine_aminotransferase = col2.number_input("Alanine_Aminotransferase (U/L)", 
                                                    min_value=None,
                                                    step=5,
                                                    value=default_vals['Alanine_Aminotransferase'],
                                                    help="The normal range is 4 to 36 U/L.")
        
        aspartate_aminotransferase = col1.number_input("Aspartate_Aminotransferase (U/L)", 
                                                    min_value=None,
                                                    step=5,
                                                    value=default_vals['Aspartate_Aminotransferase'],
                                                    help="The normal range is 8 to 33 U/L.")
        
        total_proteins = col2.number_input("Total_Proteins (g/dL)", 
                                        min_value=None,
                                        step=0.5,
                                        value=default_vals['Total_Proteins'],
                                        help="The normal range is 6.0 to 8.3 grams per deciliter (g/dL) or 60 to 83 g/L.")
        
        albumin = col1.number_input("Albumin (G/dL)", 
                                min_value=None,
                                step=0.5,
                                value=default_vals['Albumin'],
                                help="The normal range is 3.4 to 5.4 g/dL (34 to 54 g/L).")
        
        albumin_and_globulin_ratio = col2.number_input("Albumin_and_Globulin_Ratio", 
                                                    min_value=None,
                                                    step=0.2,
                                                    value=default_vals['Albumin_and_Globulin_Ratio'],
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
        
    def form_functions(self, model):
        with st.form("my_form"):
        
            get_values = self.input_data_fields()
            
            submitted = st.form_submit_button("Submit", type="primary")
            if submitted:
                data_values = pd.DataFrame([get_values])
                
                # Get predictions
                with st.spinner('Making prediction...'):
                    time.sleep(3)

                prediction = model.predict(data_values)
                print("Prediction: ", prediction[0])

                prediction_msg = "The supplied values suggest that the patient does not have a liver disease." if prediction == 0 else "The supplied values suggest that the patient has a liver disease. It is suggested to provide critical emphasis on diagnosing further symptoms of the patient. "
        
                st.subheader("Diagnosis:")

                if prediction == 0:
                    print("Success")
                    st.success(prediction_msg)

                else:
                    st.error(prediction_msg)
    
    def sample_data(self, df):

        test_data = df.drop('Diagnosis', axis=1).to_dict(orient='records')
        return test_data

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
                </style><div class="footer"><p>Developed by <a style='display: block; text-align: center;' target="_blank">Merishna S. Suwal for DSCI-6003 Final Project</a></p></div>
                """

        return footer
        
