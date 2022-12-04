import streamlit as st
import pandas as pd
from util import Util
import time

util = Util(file_path='./data/indian_liver_patient.csv')
st.header("LIVER DISEASE PREDICTION APPLICATION")

# Create a text element and let the reader know the data is loading.
data_load_state = st.info('Loading data...')
# Load 10,000 rows of data into the dataframe.
X_train, X_test, y_train, y_test = util.get_data()

#train model
data_load_state.info("Training the model..")
model = util.build_model(X_train, y_train)

# Notify the reader that the data was successfully loaded.
data_load_state.info('Application is ready for predictions.')

st.subheader("Fill in your patient data here for diagnosis")

with st.form("my_form"):
    # data_dict = util.input_data_fields()
    col1, col2 = st.columns(2)
    age = col1.text_input("Age", 72)
    gender = col2.selectbox('Gender',('Male', 'Female'))
    total_bilirubin = col1.text_input("Total_Bilirubin (mg/dL)", 
                                        0.7, 
                                        help="It is normal to have some bilirubin in the blood. A normal level is: 0.1 to 1.2 mg/dL (1.71 to 20.5 µmol/L)")
    direct_bilirubin = col2.text_input("Direct_Bilirubin (mg/dL)", 
                                        0.1, 
                                        help="Normal level for Direct (also called conjugated) bilirubin is less than 0.3 mg/dL.")
    alkaline_phosphotase = col1.text_input("Alkaline_Phosphotase (IU/L)", 
                                           "182",
                                           help="The normal range is 44 to 147 international units per liter (IU/L).")
    alanine_aminotransferase = col2.text_input("Alanine_Aminotransferase (U/L)", 
                                                24,
                                                help="The normal range is 4 to 36 U/L.")
    aspartate_aminotransferase = col1.text_input("Aspartate_Aminotransferase (U/L)", 
                                                 19,
                                                 help="The normal range is 8 to 33 U/L.")
    total_proteins = col2.text_input("Total_Proteins (g/dL)", 
                                     8.9,
                                     help="The normal range is 6.0 to 8.3 grams per deciliter (g/dL) or 60 to 83 g/L.")
    albumin = col1.text_input("Albumin (G/dL)", 
                              4.9,
                              help="The normal range is 3.4 to 5.4 g/dL (34 to 54 g/L).")
    albumin_and_globulin_ratio = col2.text_input("Albumin_and_Globulin_Ratio", 
                                                 1.20,
                                                 help="The normal range for albumin/globulin ratio is over 1 , usually around 1 to 2.")


    gender = 0 if gender == "Male" else 1
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        data_values = pd.DataFrame([{'age': age, 
            'gender': gender, 
            'total_bilirubin': total_bilirubin, 
            'direct_bilirubin': direct_bilirubin, 
            'alkaline_phosphotase': alkaline_phosphotase, 
            'alanine_aminotransferase': alanine_aminotransferase, 
            'aspartate_aminotransferase': aspartate_aminotransferase, 
            'total_proteins': total_proteins, 
            'albumin': albumin, 
            'albumin_and_globulin_ratio': albumin_and_globulin_ratio}])
        
        # Get predictions
        with st.spinner('Making prediction...'):
            time.sleep(3)

        prediction = model.predict(data_values)

        prediction = "No liver disease" if prediction == 0 else "Liver disease"
 
        st.subheader("Diagnosis:")

        if prediction == 0:
            st.success(prediction)

        else:
            st.error(prediction)

st.markdown(util.page_footer(),unsafe_allow_html=True)

