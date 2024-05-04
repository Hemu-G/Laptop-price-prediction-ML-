import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Path
pickle_file_path1 =  "give path of df.pikl location"
pickle_file_path2 =  "give path of pipe.pkl location"

# Open the pickle file and load its contents into a Python object
with open(pickle_file_path1, 'rb') as file:
    df = pickle.load(file)

with open(pickle_file_path2, 'rb') as file:
    pipe = pickle.load(file)

# Define the Streamlit app
def main():
    st.title('Laptop Price Predictor')

    # Add input widgets for user input
    company = st.selectbox('Brand', df['Company'].unique())
    type_name = st.selectbox('Type', df['TypeName'].unique())
    inches = st.selectbox('Inches', df['Inches'].unique())
    ram = st.selectbox('RAM (in GB)', sorted(df['Ram'].unique()))
    weight = st.selectbox('Weight',df['Weight'].unique())
    memory_size = st.selectbox('Memory Size', df['Memory_Size'].unique())
    memory_type = st.selectbox('Memory Type', df['Memory_Type'].unique())
    touchscreen = st.selectbox('Touchscreen', df['Touchscreen'].unique())
    ips_panel = st.selectbox('IPS Panel', df['IPS Panel'].unique())
    #ppi = st.selectbox('PPI',df['PPI'].unique())
    cpu_brand = st.selectbox('CPU Brand', df['CPU Brand'].unique())
    gpu_brand = st.selectbox('GPU Brand', df['Gpu Brand'].unique())
    os = st.selectbox('OS', df['Osy'].unique())
    resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160',
                                    '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

    if st.button('Predict Price'):
        # Prepare the input data

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5/inches
        query = pd.DataFrame({
            'Company': [company],
            'TypeName': [type_name],
            'Inches': [inches],
            'Ram': [ram],
            'Weight': [weight],
            'Memory_Size': [memory_size],
            'Memory_Type': [memory_type],
            'Touchscreen': [touchscreen],
            'IPS Panel': [ips_panel],
            'PPI': [ppi],
            'CPU Brand': [cpu_brand],
            'Gpu Brand': [gpu_brand],
            'Osy': [os]
        })

        # Predict the price
        predicted_price = np.exp(pipe.predict(query))
        st.write(f'Predicted Price: INR {predicted_price[0]:,.2f}')

# Run the Streamlit app
if __name__ == '__main__':
    main()
