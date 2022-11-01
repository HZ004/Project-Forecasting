# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
# import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import streamlit as st

from tvDatafeed import TvDatafeed ,Interval


import fbprophet
from fbprophet import Prophet

# %matplotlib inline

st.title('Model Deployment: Forecasting')

st.sidebar.header('Input Company symbol listed on NSE')

COMPANY = st.sidebar.text_input("Insert Company name in Upper cases")

tv = TvDatafeed()
data = tv.get_hist(symbol=COMPANY,exchange='NSE',n_bars=5000)
data['date'] = data.index.astype(str)
new = data['date'].str.split(' ',expand=True)
data['date'] = new[0]
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

"""#FB PROPHET
"""

data2 = data
data2['ds'] = pd.to_datetime(data.index)
data2['y'] = (data2['close'])
data2 = data2[['ds','y']].reset_index(drop = True)



model = Prophet()
model.fit(data2)


future = model.make_future_dataframe(periods = 730)


pred = model.predict(future)

st.subheader('Predicted Result')
st.pyplot(model.plot(pred))

st.subheader('Other Components of FBPROPHET')
st.pyplot(model.plot_components(pred))

se = np.square(pred.loc[:, 'yhat'] - data2.y)
mse = np.mean(se)
rmse = np.sqrt(mse)
st.subheader('Root Mean Squared Error')
st.write(rmse)


