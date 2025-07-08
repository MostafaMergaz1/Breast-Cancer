import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
st.title('Breast Cancer')
df=pd.read_csv(r'C:\Users\ALMot7da\Downloads\breastcancer.csv')
encoder=LabelEncoder()
df['diagnosis']=encoder.fit_transform(df['diagnosis'])

x=df.drop(['diagnosis','fractal_dimension_mean','texture_se','smoothness_se','symmetry_se','id'],axis=1)
y=df['diagnosis']
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns=df.drop(['diagnosis','fractal_dimension_mean','texture_se','smoothness_se','symmetry_se','id'], axis=1).columns)


st.sidebar.header('User Input')
def user_input():
    radius_mean = st.sidebar.slider('radius_mean', -2.0, 4.0, 0.0)
    texture_mean = st.sidebar.slider('texture_mean', -2.0, 5.0, 0.0)
    perimeter_mean = st.sidebar.slider('perimeter_mean', -2.0, 4.0, 0.0)
    area_mean = st.sidebar.slider('area_mean', -1.5, 5.5, 0.0)
    smoothness_mean = st.sidebar.slider('smoothness_mean', -3.0, 4.5, 0.0)
    compactness_mean = st.sidebar.slider('compactness_mean', -1.5, 4.5, 0.0)
    concavity_mean = st.sidebar.slider('concavity_mean', -1.2, 4.3, 0.0)
    concave_points_mean = st.sidebar.slider('concave points_mean', -1.3, 3.9, 0.0)
    symmetry_mean = st.sidebar.slider('symmetry_mean', -2.7, 4.5, 0.0)
    radius_se = st.sidebar.slider('radius_se', -1.1, 8.9, 0.0)
    perimeter_se = st.sidebar.slider('perimeter_se', -1.1, 5.7, 0.0)
    area_se = st.sidebar.slider('area_se', -1.3, 6.2, 0.0)
    compactness_se = st.sidebar.slider('compactness_se', -1.4, 3.4, 0.0)
    concavity_se = st.sidebar.slider('concavity_se', -1.3, 4.7, 0.0)
    concave_points_se = st.sidebar.slider('concave points_se', -1.7, 2.7, 0.0)
    fractal_dimension_se = st.sidebar.slider('fractal_dimension_se', -1.6, 6.8, 0.0)
    radius_worst = st.sidebar.slider('radius_worst', -1.7, 4.1, 0.0)
    texture_worst = st.sidebar.slider('texture_worst', -2.2, 3.9, 0.0)
    perimeter_worst = st.sidebar.slider('perimeter_worst', -1.7, 4.3, 0.0)
    area_worst = st.sidebar.slider('area_worst', -1.2, 5.9, 0.0)
    smoothness_worst = st.sidebar.slider('smoothness_worst', -2.7, 4.0, 0.0)
    compactness_worst = st.sidebar.slider('compactness_worst', -1.5, 5.1, 0.0)
    concavity_worst = st.sidebar.slider('concavity_worst', -1.3, 4.7, 0.0)
    concave_points_worst = st.sidebar.slider('concave points_worst', -1.7, 2.7, 0.0)
    symmetry_worst = st.sidebar.slider('symmetry_worst', -2.2, 6.0, 0.0)
    fractal_dimension_worst = st.sidebar.slider('fractal_dimension_worst', -1.6, 6.8, 0.0)

    df = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'radius_se': radius_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,
        'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst
    }

    features = pd.DataFrame([df])
    return features

input_df = user_input()
st.header(
    """
    Predict Clicked ON ADD Based on Data Features
    """
)    

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
#pred_t=model.predict(x_train)
pred_s=model.predict(x_test)
if st.button('Predict'):
    input_df = input_df[x_train.columns]
    
    prediction = model.predict(input_df)[0]
    result = '(Benign)' if prediction == 1 else '(Malignant)'
    
    st.write(f'Model Prediction: {result}')
