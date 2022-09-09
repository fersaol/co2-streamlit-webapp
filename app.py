import streamlit as st
import sys
import os
from PIL import Image
import datetime
sys.path.append(os.getcwd() + "\\model")
sys.path.append(os.getcwd() + "\\utils")
from model import modelo_final_concampos as mymodel
from utils import general_purpose as gps


# App Name
st.title("Environmental Energy Production Checker")
st.header("This tool helps countries to monitor and adjust its energy production policies")
st.write("The tool uses Machine Learning Algortihms to analize the energy production mix")

image = Image.open("chimeneas.jpg")
st.image(image,use_column_width=True)

st.write("Please enter your data below:")

Country = st.text_input(label="Enter your country name",key="country_input")

Year= st.date_input(label="Autocompletion field, date in format: yyyy-mm-dd",
                    value=datetime.datetime.today(),
                    key="date_input",disabled=True)

GDP = st.number_input(label="Enter the GDP of your country in PPP(base:2015)",
                key="GDP_input",min_value=0)

Population= st.number_input(label="Enter the Population of your Country",
                key="Population_input",min_value=0)

Energy_production= st.number_input(label="Enter the Amount of Energy Production",
                key="Energy_production_input")

Energy_consumption= st.number_input(label="Enter the Amount of Energy Consumption",
                key="Energy_consumption_input")

CO2_emission= st.number_input(label="Enter the CO2 Emissions of your Country",
                key="co2_input")

energy_type= st.number_input(label="Enter the code -> 0: renewables, 1: nuclear,\
                                    2: gas, 3: petroleum, 4: coal",
                            key="energy_type_input",
                            min_value=0,max_value=4)

prediction = st.button(label="Predict and Classifies",key="Prediction_button",)

if prediction:
    pred = mymodel.Final_Model(Country,Year,GDP,Population,Energy_production,
                Energy_consumption,CO2_emission,energy_type).run_whole_model()
    
    resultado = st.write(pred)

