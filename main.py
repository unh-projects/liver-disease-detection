import streamlit as st
import pandas as pd
from util import Util
import time

st.set_page_config(
        page_title="Liver Disease Prediction",
)

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


## FORM for Prediction
st.subheader("Fill in your patient data here for diagnosis")

with st.form("my_form"):
 
    get_values = util.input_data_fields()
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        data_values = pd.DataFrame([get_values])
        
        # Get predictions
        with st.spinner('Making prediction...'):
            time.sleep(3)

        print("DATA values: ", data_values)
        prediction = model.predict(data_values)
        print("Prediction: ", prediction[0])

        prediction_msg = "No liver disease" if prediction == 0 else "Liver disease"
 
        st.subheader("Diagnosis:")

        if prediction == 0:
            print("Success")
            st.success(prediction_msg)

        else:
            st.error(prediction_msg)

st.markdown(util.page_footer(),unsafe_allow_html=True)

