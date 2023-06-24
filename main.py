import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

st.title("RAINFALL DATASET OF INDIA")

#importing data set
district_wise_rainfall = pd.read_csv(r"C:\Users\ADMIN\Downloads\rainfall data set with year\district wise rainfall normal.csv")
year_wise_rainfall = pd.read_csv(r"C:\Users\ADMIN\PycharmProjects\rainfall prediction streamlit\rainfall in india 1901-2015.csv")

#replaceing null values
def null_val(data,clm):
    Mean = data[clm].mean()
    data[clm] = data[clm].fillna(Mean)

colm = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
for clm in colm:
    null_val(year_wise_rainfall,clm)

#converting dats set for our use
groups = year_wise_rainfall.groupby('SUBDIVISION')['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']   #year_wise_rainfall1 = groups.get_group(('TAMIL NADU'))
OBSERVED_CENTER = st.selectbox("Observed Centers in INDIA",['ANDAMAN & NICOBAR ISLANDS', 'ARUNACHAL PRADESH','ASSAM & MEGHALAYA', 'NAGA MANI MIZO TRIPURA','SUB HIMALAYAN WEST BENGAL & SIKKIM', 'GANGETIC WEST BENGAL','ORISSA', 'JHARKHAND', 'BIHAR', 'EAST UTTAR PRADESH','WEST UTTAR PRADESH', 'UTTARAKHAND', 'HARYANA DELHI & CHANDIGARH','PUNJAB', 'HIMACHAL PRADESH', 'JAMMU & KASHMIR', 'WEST RAJASTHAN','EAST RAJASTHAN', 'WEST MADHYA PRADESH', 'EAST MADHYA PRADESH','GUJARAT REGION', 'SAURASHTRA & KUTCH', 'KONKAN & GOA','MADHYA MAHARASHTRA', 'MATATHWADA', 'VIDARBHA', 'CHHATTISGARH','COASTAL ANDHRA PRADESH', 'TELANGANA', 'RAYALSEEMA', 'TAMIL NADU','COASTAL KARNATAKA', 'NORTH INTERIOR KARNATAKA','SOUTH INTERIOR KARNATAKA', 'KERALA', 'LAKSHADWEEP'])
#while choosing observed centre depends upon users wish:
collected_data = groups.get_group((OBSERVED_CENTER))
year_order_data  = collected_data.melt(['YEAR']).reset_index()
final_data = year_order_data[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])
#renaming features
final_data.columns=['INDEX','YEAR','Month','avg_rainfall']
#converting obj to int on Month feature:
d={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
final_data['Month']=final_data['Month'].map(d)
final_data['Date']=pd.to_datetime(final_data.assign(Day=1).loc[:,['YEAR','Month','Day']])
cols=['avg_rainfall']
avg_rain=final_data[cols]
series=avg_rain
data_raw = series.values.astype("float32")
scaler = MinMaxScaler(feature_range = (0, 1))
data_set = scaler.fit_transform(data_raw)

nav = st.sidebar.radio("Navigation", ["Prediction","Monsoon in India"])

#prediction:
if nav == "Prediction":
    st.header("RAINFALL PREDICTION")
    #Model building
    x = final_data[["YEAR", "Month"]]
    y = final_data.iloc[:, 3].values
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    ##prdiction area
    year = st.number_input("Enter the year",min_value=2016,max_value=2030,step=1)
    month = st.number_input("Enter the month number",min_value=1,max_value=12,step=1)
    user_input = {'YEAR': [year], 'Month': [month]}
    pred_input = pd.DataFrame(user_input)
    pred = lin_reg.predict(pred_input)[0]

    if st.button("predict"):
        st.write("Your predicted average rainfall for Year",year,"and month",month,"is",pred)

#MONSOON REGINON:
groups1 = year_wise_rainfall.groupby('SUBDIVISION')['YEAR','Jun-Sep']
groups2 = year_wise_rainfall.groupby('SUBDIVISION')['YEAR','Oct-Dec']

if nav == "Monsoon in India":
    st.header("Monsoon in India")
    Types_of_monsoon = st.selectbox("Monsoon Types",['SouthWest Monsoon', 'NorthEast Monsoon'])
    if Types_of_monsoon == 'SouthWest Monsoon':
        Southwest = st.selectbox('SW monsoon region',
                                 ['KERALA', 'COASTAL KARNATAKA', 'SOUTH INTERIOR KARNATAKA', 'KONKAN & GOA',
                                  'MADHYA MAHARASHTRA'])
        swdf1 = groups1.get_group((Southwest))
        swdf1 = swdf1.melt(['YEAR']).reset_index()
        #st.table(swdf1)
        #prediction
        x = (swdf1.iloc[:,1].values).reshape(-1,1)
        y = swdf1.iloc[:, 3].values
        lin_reg = LinearRegression()
        lin_reg.fit(x, y)
        #getting users input
        year = st.number_input("Enter the year", min_value=2016, max_value=2030, step=1)
        sw_user_input = {'YEAR': [year]}
        sw_pred_input = pd.DataFrame(sw_user_input)
        sw_pred = lin_reg.predict(sw_pred_input)[0]
        if st.button("Predict"):
            st.write("Your predicted average rainfall of SouthWest Monsoon for State",Southwest,'on Year', year, "is", sw_pred)
    if Types_of_monsoon == 'NorthEast Monsoon':
        Northeast = st.selectbox('NE monsoon region',
                                 ['TAMIL NADU', 'KERALA', 'NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA',
                                  'TELANGANA', 'COASTAL ANDHRA PRADESH'])
        nedf1 = groups2.get_group((Northeast))
        nedf1 = nedf1.melt(['YEAR']).reset_index()
        st.table(nedf1)
        # prediction
        x = (nedf1.iloc[:, 1].values).reshape(-1, 1)
        y = nedf1.iloc[:, 3].values
        lin_reg = LinearRegression()
        lin_reg.fit(x, y)
        # getting users input
        year = st.number_input("Enter the year", min_value=2016, max_value=2030, step=1)
        ne_user_input = {'YEAR': [year]}
        ne_pred_input = pd.DataFrame(ne_user_input)
        ne_pred = lin_reg.predict(ne_pred_input)[0]
        if st.button("Predict"):
            st.write("Your predicted average rainfall of SouthWest Monsoon for State", Northeast, 'on Year', year, "is",
                     ne_pred)
