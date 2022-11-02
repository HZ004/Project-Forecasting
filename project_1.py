# -*- coding: utf-8 -*-
#%matplotlib inline

import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt
import cufflinks as cf
import warnings
warnings.filterwarnings("ignore")
import pmdarima as pm
import streamlit as st
from tvDatafeed import TvDatafeed ,Interval
import fbprophet
from fbprophet import Prophet



st.title('Model Deployment: Forecasting')
st.sidebar.header('Input Company symbol listed on NSE')

COMPANY = st.sidebar.text_input("Insert Company name in Upper cases")
MODEL = st.sidebar.selectbox('Forecasting Model',('Model Based','Data Driven','ARIMA','LSTM Artificial Neural Network','FB Prophet'))

tv = TvDatafeed()
data = tv.get_hist(symbol=COMPANY,exchange='NSE',n_bars=5000)
data['date'] = data.index.astype(str)
new = data['date'].str.split(' ',expand=True)
data['date'] = new[0]
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

timeseriesdf = data[['close']]
timeseriessq = data['close']

st.subheader('Candlestick Chart')
fig = cf.Figure(data=[cf.Candlestick(x=data.index, 
			       open=data['open'],
			       high = data['high'],
			       low = data['low'],
			       close = data['close'])])
fig.update_layout(xaxis_rangeslider_visible=False)
st.write(fig)


st.subheader('Line Chart')
fig2 = plt.figure(figsize = (20,8))
plt.plot(data.close)
st.write(fig2)

##################################################################################
def model(var):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=var,exchange='NSE',n_bars=5000)
    data['date'] = data.index.astype(str)
    new = data['date'].str.split(' ',expand=True)
    data['date'] = new[0]
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date',drop=False)
    
    heatmapdata = data[['date','close']]
    heatmapdata['date'] = pd.to_datetime(heatmapdata['date']) 
	# Extracting Day, weekday name, month name, year from the Date column using 
	# Date functions from pandas 
    heatmapdata["month"] = heatmapdata['date'].dt.strftime("%b") # month extraction
    heatmapdata["year"] = heatmapdata['date'].dt.strftime("%Y") # year extraction
    heatmapdata["Day"] = heatmapdata['date'].dt.strftime("%d") # Day extraction
    heatmapdata["wkday"] = heatmapdata['date'].dt.strftime("%A") # weekday extraction

    heatmap_y_month = pd.pivot_table(data = heatmapdata,
                        values = "close",
                        index = "year",
                        columns = "month",
                        aggfunc = "mean",
                        fill_value=0)
    heatmap_y_month1 = heatmap_y_month[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]

    st.header('Model Based Forecast Result')
    st.subheader('Heatmap')
    fig = plt.figure(figsize=(20,10))
    sns.heatmap(heatmap_y_month1,annot=True,fmt="g",cmap = 'YlOrBr')
    st.pyplot(fig)

    # Boxplot for every month
    st.subheader('Monthly Boxplot')
    fig = plt.figure(figsize=(20,10))
    sns.boxplot(x="month",y="close",data=heatmapdata, order = ["Jan", "Feb","Mar", "Apr","May", "Jun","Jul", "Aug","Sep", "Oct","Nov", "Dec"])
    st.pyplot(fig)

    st.subheader('Yearly Boxplot')
    fig = plt.figure(figsize=(20,10))
    sns.boxplot(x="year",y="close",data=heatmapdata)
    sns.lineplot(x="year",y="close",data=heatmapdata)
    st.pyplot(fig)
	
    sns.lineplot(x="year",y="close",data=heatmapdata)

    #### Splitting data
    data1 = heatmapdata

    data1['t'] = np.arange(1,data1.shape[0]+1)
    data1['t_square'] = np.square(data1.t)
    data1['log_close'] = np.log(data1.close)
    data2 = pd.get_dummies(data1['month'])
    data1 = pd.concat([data1, data2],axis=1)
    data1 = data1.reset_index(drop = True)
    
    # Using 3/4th data for training and remaining for testing
    test_size = round(0.25 * (data1.shape[0]+1))

    Train = data1[:-test_size]
    Test = data1[-test_size:]
    
    ## Trying basic models

    #Linear Model
    import statsmodels.formula.api as smf 

    linear_model = smf.ols('close~t',data=Train).fit()
    pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
    rmse_linear = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_linear))**2))

    #Exponential
    Exp = smf.ols('log_close~t',data=Train).fit()
    pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
    rmse_Exp = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Exp)))**2))

    #Quadratic 
    Quad = smf.ols('close~t+t_square',data=Train).fit()
    pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
    rmse_Quad = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_Quad))**2))

    #Additive seasonality 
    add_sea = smf.ols('close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
    pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
    rmse_add_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_add_sea))**2))

    #Additive Seasonality Quadratic 
    add_sea_Quad = smf.ols('close~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
    pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
    rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_add_sea_quad))**2))

    ##Multiplicative Seasonality
    Mul_sea = smf.ols('log_close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
    pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
    rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Mult_sea)))**2))

    #Multiplicative Additive Seasonality 
    Mul_Add_sea = smf.ols('log_close~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
    pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
    rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Mult_add_sea)))**2))

    #Compare the results 
    datamodel = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),
            "RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
    table_rmse=pd.DataFrame(datamodel)
    table = table_rmse.sort_values(['RMSE_Values'],ignore_index = True)

    bestmodel = table.iloc[0,0]

    if bestmodel == "rmse_linear" :
        formula = 'close~t'

    if bestmodel == "rmse_Exp":
        formula = 'log_close~t'

    if bestmodel == "rmse_Quad" :
        formula = 'close~t+t_square'

    if bestmodel == "rmse_add_sea":
        formula = 'close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'

    if bestmodel == "rmse_add_sea_quad":
        formula = 'close~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'

    if bestmodel == "rmse_Mult_sea":
        formula = 'log_close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'

    if bestmodel == "rmse_Mult_add_sea":
        formula = 'log_close~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'


    #Build the model on entire data set
    model_full = smf.ols(formula,data=data1).fit()
    pred_new  = pd.Series(model_full.predict(data1))

    if bestmodel == ("rmse_Exp" or "rmse_Mult_sea" or "rmse_Mult_add_sea"):
        data1["forecasted_close"] = pd.Series(np.exp(pred_new))
    else:
        data1["forecasted_close"] = pd.Series((pred_new))

    st.subheader('Best Basic Mathematical Model')
    fig = plt.figure(figsize = (20,8))
    plt.plot(data1[['close','forecasted_close']].reset_index(drop=True))
    st.pyplot(fig)
	
######################################################################################

def datad(var):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=var,exchange='NSE',n_bars=5000)
    data['date'] = data.index.astype(str)
    new = data['date'].str.split(' ',expand=True)
    data['date'] = new[0]
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date',drop=False)
    
    import statsmodels.graphics.tsaplots as tsa_plots
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
    from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    heatmapdata = data[['date','close']]

    heatmapdata['date'] = pd.to_datetime(heatmapdata['date']) 
    # Extracting Day, weekday name, month name, year from the Date column using Date functions from pandas 
    heatmapdata["month"] = heatmapdata['date'].dt.strftime("%b") # month extraction
    heatmapdata["year"] = heatmapdata['date'].dt.strftime("%Y") # year extraction
    heatmapdata["Day"] = heatmapdata['date'].dt.strftime("%d") # Day extraction
    heatmapdata["wkday"] = heatmapdata['date'].dt.strftime("%A") # weekday extraction

    data1 = heatmapdata

    data1['t'] = np.arange(1,data1.shape[0]+1)
    data1['t_square'] = np.square(data1.t)
    data1['log_close'] = np.log(data1.close)
    data2 = pd.get_dummies(data1['month'])
    data1 = pd.concat([data1, data2],axis=1)
    data1 = data1.reset_index(drop = True)

    # Using 3/4th data for training and remaining for testing
    test_size = round(0.25 * (data1.shape[0]+1))
    Train = data1[:-test_size]
    Test = data1[-test_size:]

    st.header('Data Driven Forecast Result')
    st.subheader('Moving Average(MA)')
    fig = plt.figure(figsize=(20,8))
    data1['close'].plot(label="org")
    for i in range(50,201,50):
        data1["close"].rolling(i).mean().plot(label=str(i))
    plt.legend(loc='best')
    st.pyplot(fig)
	
    ### Evaluation Metric RMSE
    def RMSE(pred,org):
        MSE = np.square(np.subtract(org,pred)).mean()   
        return np.sqrt(MSE)

    ### Simple Exponential Method
    ses_model = SimpleExpSmoothing(Train["close"]).fit(smoothing_level=0.2)
    pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
    rmseses = RMSE(pred_ses,Test.close)
    
    ### Holt method
    hw_model = Holt(Train["close"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
    pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
    rmsehw = RMSE(pred_hw,Test.close)

    ### Holts winter exponential smoothing with additive seasonality and additive trend
    hwe_model_add_add = ExponentialSmoothing(Train["close"],seasonal="add",trend="add",seasonal_periods=365).fit() #add the trend to the model
    pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
    rmsehwaa = RMSE(pred_hwe_add_add,Test.close)

    ### Holts winter exponential smoothing with multiplicative seasonality and additive trend
    hwe_model_mul_add = ExponentialSmoothing(Train["close"],seasonal="mul",trend="add",seasonal_periods=365).fit() 
    pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
    rmsehwma = RMSE(pred_hwe_mul_add,Test.close)

    ### Final Model by combining train and test
    datamodel1 = {"MODEL":pd.Series(["rmse_ses","rmse_hw","rmse_hwe_add_add","rmse_hwe_mul_add"]),"RMSE_Values":pd.Series([rmseses,rmsehw,rmsehwaa,rmsehwma])}

    table_rmse1 = pd.DataFrame(datamodel1)

    table1 = table_rmse1.sort_values(['RMSE_Values'],ignore_index = True)

    bestmodel1 = table1.iloc[0,0]

    if bestmodel1 == "rmse_hwe_add_add" :
        formula1 = ExponentialSmoothing(data["close"],seasonal="add",trend="add",seasonal_periods=365).fit()
    if bestmodel1 == "rmse_hwe_mul_add":
        formula1 = ExponentialSmoothing(data["close"],seasonal="mul",trend="add",seasonal_periods=365).fit()
    if bestmodel1 == "rmse_ses" :
        formula1 = SimpleExpSmoothing(data["close"]).fit(smoothing_level=0.2)
    if bestmodel1 == "rmse_hw":
        formula1 = Holt(data["close"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
    
    #Forecasting for next 12 time periods
    forecasted = formula1.forecast(730)

    st.subheader('Best Holt Winters Model')
    fig = plt.figure(figsize=(20,8))
    plt.plot(data1.close, label = "Actual")
    plt.plot(forecasted, label = "Forecasted")
    plt.legend()
    st.pyplot(fig)

###########################################################################################################

def arima(var):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=var,exchange='NSE',n_bars=5000)
    data['date'] = data.index.astype(str)
    new = data['date'].str.split(' ',expand=True)
    data['date'] = new[0]
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date',drop=False)
    
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import kpss
	
    st.header('Auto ARIMA Forecast Result')
    st.subheader('Determining stationarity of the dataset using Augmented Dickey-Fuller Test')

    result=adfuller (data['close'])
    st.write(print('Test Statistic: %f' %result[0]))
    st.write(print('p-value: %f' %result[1]))
    st.write(print('Critical values:'))
    for key, value in result[4].items ():
        st.write(print('\t%s: %.3f' %(key, value)))

	
    st.subheader('Determining stationarity of the dataset usingKwiatkowski Phillips Schmidt Shin (KPSS) test')
    result_kpss_ct=kpss(data['close'],regression="ct")
    st.write(print('Test Statistic: %f' %result_kpss_ct[0]))
    st.write(print('p-value: %f' %result_kpss_ct[1]))
    st.write(print('Critical values:'))
    for key, value in result_kpss_ct[3].items():
        st.write(print('\t%s: %.3f' %(key, value)))

    # Auto ARIMA on complete Dataset
    import itertools
    from math import sqrt
    import statsmodels.api as sm
    from sklearn.metrics import mean_squared_error
    from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    ARIMA_model = pm.auto_arima(data['close'], 
                        start_p=1, 
                        start_q=1,
                        test='adf', # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                        d=None,# let model determine 'd'
                        seasonal=False, # No Seasonality for standard ARIMA
                        trace=False, #logs 
                        error_action='warn', #shows errors ('ignore' silences these)
                        suppress_warnings=True,
                        stepwise=True)
    
    from pandas.tseries.frequencies import DAYS
    def forecast(ARIMA_model, periods=730):
        # Forecast
        n_periods = periods
        fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
        index_of_fc = pd.date_range(data.index[-1] + pd.DateOffset(days=1), periods = n_periods, freq='D')
        
        # make series for plotting purpose
        fitted_series = pd.Series(fitted.values, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        # Plot
	st.subheader('Auto-ARIMA Forecast')
        fig = plt.figure(figsize=(20,8))
        plt.plot(data["close"])
        plt.plot(fitted_series, color='darkgreen')
        plt.fill_between(lower_series.index, 
                        lower_series, 
                        upper_series, 
                        color='k', alpha=.15)

        plt.title("ARIMA - Forecast of Close Price")
        st.pyplot(fig)

    forecast(ARIMA_model)

#############################################################################################
'''
def lstm():
	


def fb():
	
####################################################################



databoxplot = data
databoxplot['date'] = databoxplot.index.astype(str)
new = databoxplot['date'].str.split(' ',expand=True)
databoxplot['date'] = new[0]
databoxplot = databoxplot[['date','close']]
databoxplot['year'] = databoxplot['date'].str[:4].astype(int)
databoxplot['date1'] = databoxplot['date'].str[5:]



years = pd.pivot_table(databoxplot,index = 'date1', values = 'close',columns='year', aggfunc = sum)

plt.figure(figsize = (20,8))
years.boxplot()


# create a scatter plot
plt.figure(figsize = (20,8))
pd.plotting.lag_plot(timeseriesdf)

# create an autocorrelation plot
from statsmodels.graphics.tsaplots import plot_acf

plt.rc("figure", figsize=(20,8))
plot_acf(timeseriesdf, lags=1000)
plt.show()

# Taking care of outliers


data['returns percentage'] = ((data['close']/data['close'].shift(1)) -1)*100
data['returns percentage'].hist(bins = 100, label = 'Stock', alpha = 0.5, figsize = (15,7))
plt.legend()

datac = data.iloc[1:]
x = datac['returns percentage']


plt.boxplot(x,vert=False)

datac = datac[datac['returns percentage'] < 5]
datac = datac[datac['returns percentage'] > -5]



timeseriesdf = data[['close']]
timeseriessq = data['close']



upsampled = timeseriessq.resample('H').mean()


interpolated = upsampled.interpolate(method='linear')


# downsample to quarterly intervals
resample = timeseriessq.resample('Q')
downsampled = resample.mean()



# load and plot a time series


dataframe = pd.DataFrame(np.sqrt(timeseriesdf.values), columns = ['close'])


dataframe = pd.DataFrame(np.log(timeseriesdf.values), columns = ['close'])

# Forecasting - Model Based"""

heatmapdata = data[['date','close']]
heatmapdata['date'] = pd.to_datetime(heatmapdata['date'])


# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

heatmapdata["month"] = heatmapdata['date'].dt.strftime("%b") # month extraction
heatmapdata["year"] = heatmapdata['date'].dt.strftime("%Y") # year extraction
heatmapdata["Day"] = heatmapdata['date'].dt.strftime("%d") # Day extraction
heatmapdata["wkday"] = heatmapdata['date'].dt.strftime("%A") # weekday extraction

heatmap_y_month = pd.pivot_table(data = heatmapdata,
                                 values = "close",
                                 index = "year",
                                 columns = "month",
                                 aggfunc = "mean",
                                 fill_value=0)
heatmap_y_month1 = heatmap_y_month[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]


plt.figure(figsize=(20,10))
modelplot0 = sns.heatmap(heatmap_y_month1,
             annot=True,
             fmt="g",
             cmap = 'YlOrBr')

# Boxplot for every
plt.figure(figsize=(20,10))
modelplot1 = sns.boxplot(x="month",y="close",data=heatmapdata, order = ["Jan", "Feb","Mar", "Apr","May", "Jun","Jul", "Aug","Sep", "Oct","Nov", "Dec"])

plt.figure(figsize=(20,10))
modelplot2 = sns.boxplot(x="year",y="close",data=heatmapdata)

sns.lineplot(x="year",y="close",data=heatmapdata)

#### Splitting data"""

data1 = heatmapdata


data1['t'] = np.arange(1,data1.shape[0]+1)
data1['t_square'] = np.square(data1.t)
data1['log_close'] = np.log(data1.close)
data2 = pd.get_dummies(data1['month'])
data1 = pd.concat([data1, data2],axis=1)
data1 = data1.reset_index(drop = True)


# Using 3/4th data for training and remaining for testing
test_size = round(0.25 * (data1.shape[0]+1))

Train = data1[:-test_size]
Test = data1[-test_size:]


## Trying basic models"""

#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('close~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_linear))**2))

#Exponential
Exp = smf.ols('log_close~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Exp)))**2))


#Quadratic 
Quad = smf.ols('close~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_Quad))**2))


#Additive seasonality 
add_sea = smf.ols('close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_add_sea))**2))


#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('close~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_add_sea_quad))**2))


##Multiplicative Seasonality
Mul_sea = smf.ols('log_close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Mult_sea)))**2))


#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_close~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Mult_add_sea)))**2))


#Compare the results 
datamodel = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(datamodel)
table = table_rmse.sort_values(['RMSE_Values'],ignore_index = True)


bestmodel = table.iloc[0,0]

if bestmodel == "rmse_linear" :
  formula = 'close~t'

if bestmodel == "rmse_Exp":
  formula = 'log_close~t'

if bestmodel == "rmse_Quad" :
  formula = 'close~t+t_square'

if bestmodel == "rmse_add_sea":
  formula = 'close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'

if bestmodel == "rmse_add_sea_quad":
  formula = 'close~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'

if bestmodel == "rmse_Mult_sea":
  formula = 'log_close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'

if bestmodel == "rmse_Mult_add_sea":
  formula = 'log_close~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'


#Build the model on entire data set
model_full = smf.ols(formula,data=data1).fit()

pred_new  = pd.Series(model_full.predict(data1))


if bestmodel == ("rmse_Exp" or "rmse_Mult_sea" or "rmse_Mult_add_sea"):
  data1["forecasted_close"] = pd.Series(np.exp(pred_new))
else:
  data1["forecasted_close"] = pd.Series((pred_new))

plt.figure(figsize = (20,8))
modelplot3 = plt.plot(data1[['close','forecasted_close']].reset_index(drop=True))

################################################################################
# Forecasting - Data Driven

import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#### Moving Average

x=20

plt.figure(figsize=(24,7))
data1['close'].plot(label="org")
data1["close"].rolling(x).mean().plot(label=str(x))
plt.legend(loc='best')

fig = plt.figure(figsize=(24,7))
data1['close'].plot(label="org")
for i in range(50,201,50):
    data1["close"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')

dataplot0 = fig
#### Time series decomposition plot 

decompose_ts_add = seasonal_decompose(data1['close'], period = 365)
decompose_ts_add.plot()

plt.show()

#### ACF plots and PACF plots

dataplot1 = tsa_plots.plot_acf(data1.close,lags=350)
tsa_plots.plot_pacf(data1.close,lags=350)
plt.show()

### Evaluation Metric RMSE

def RMSE(pred,org):
  MSE = np.square(np.subtract(org,pred)).mean()   
  return np.sqrt(MSE)

### Simple Exponential Method



ses_model = SimpleExpSmoothing(Train["close"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
rmseses = RMSE(pred_ses,Test.close)

### Holt method

# Holt method 
hw_model = Holt(Train["close"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
rmsehw = RMSE(pred_hw,Test.close)

### Holts winter exponential smoothing with additive seasonality and additive trend

hwe_model_add_add = ExponentialSmoothing(Train["close"],seasonal="add",trend="add",seasonal_periods=365).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
rmsehwaa = RMSE(pred_hwe_add_add,Test.close)

### Holts winter exponential smoothing with multiplicative seasonality and additive trend

hwe_model_mul_add = ExponentialSmoothing(Train["close"],seasonal="mul",trend="add",seasonal_periods=365).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
rmsehwma = RMSE(pred_hwe_mul_add,Test.close)

### Final Model by combining train and test

#Compare the results 
datamodel1 = {"MODEL":pd.Series(["rmse_ses","rmse_hw","rmse_hwe_add_add","rmse_hwe_mul_add"]),"RMSE_Values":pd.Series([rmseses,rmsehw,rmsehwaa,rmsehwma])}
table_rmse1 = pd.DataFrame(datamodel1)
table1 = table_rmse1.sort_values(['RMSE_Values'],ignore_index = True)


bestmodel1 = table1.iloc[0,0]


if bestmodel1 == "rmse_hwe_add_add" :
  formula1 = ExponentialSmoothing(data["close"],seasonal="add",trend="add",seasonal_periods=365).fit()

if bestmodel1 == "rmse_hwe_mul_add":
  formula1 = ExponentialSmoothing(data["close"],seasonal="mul",trend="add",seasonal_periods=365).fit()

if bestmodel1 == "rmse_ses" :
  formula1 = SimpleExpSmoothing(data["close"]).fit(smoothing_level=0.2)

if bestmodel1 == "rmse_hw":
  formula1 = Holt(data["close"]).fit(smoothing_level=0.8, smoothing_slope=0.2)

#Forecasting for next 12 time periods
forecasted = formula1.forecast(730)

dataplot2 = plt.figure(figsize=(24,7))
plt.plot(data1.close, label = "Actual")
plt.plot(forecasted, label = "Forecasted")
plt.legend()
#########################################################################
# Forecasting using ARIMA,SARIMA and SARIMAX model

### Checking if the data is stationary or not


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

fig=plt.figure(figsize=(20,8))
sns.lineplot(data=timeseriesdf,x='date',y='close')
plt.tick_params(
    axis='x',        
    which='both',   
    bottom=False,      
    top=False,        
    labelbottom=False) 
plt.show()

### Augmented Dickey-Fuller Test

result=adfuller (timeseriesdf['close'])
print('Test Statistic: %f' %result[0])
print('p-value: %f' %result[1])
print('Critical values:')
for key, value in result[4].items ():
     print('\t%s: %.3f' %(key, value))

### Kwiatkowski Phillips Schmidt Shin (KPSS) test

result_kpss_ct=kpss(timeseriesdf['close'],regression="ct")
print('Test Statistic: %f' %result_kpss_ct[0])
print('p-value: %f' %result_kpss_ct[1])
print('Critical values:')
for key, value in result_kpss_ct[3].items():
     print('\t%s: %.3f' %(key, value))

## Test Statistic in both Tests is greater than standard 0.05
## Hence the data can be classified as not Stationary
#Transforming of the data

df_log=np.sqrt(timeseriesdf['close'])
df_diff=df_log.diff().dropna()
result=adfuller (df_diff)
print('Test Statistic: %f' %result[0])
print('p-value: %f' %result[1])
print('Critical values:')
for key, value in result[4].items ():
     print('\t%s: %.3f' %(key, value))

result_kpss_ct_log=kpss(df_diff,regression="ct")
print('Test Statistic: %f' % np.round(result_kpss_ct_log[0],2))
print('p-value: %f' %result_kpss_ct_log[1])
print('Critical values:')
for key, value in result_kpss_ct_log[3].items():
     print('\t%s: %.3f' %(key, value))


fig=plt.figure(figsize=(20,8))
sns.lineplot(data=df_diff)
plt.tick_params(
    axis='x',        
    which='both',   
    bottom=False,      
    top=False,        
    labelbottom=False) 
plt.show()

## Values of both the test statistics is below the standard value of 0.05

plt.figure(figsize=(20,8))
plt.plot(df_diff,label="after")
plt.plot(timeseriesdf,label="before")
plt.tick_params(
    axis='x',        
    which='both',   
    bottom=False,      
    top=False,        
    labelbottom=False)
plt.legend()
plt.show()


stationarydf1 = df_diff

## We can also use basic differencing method to make data stationary

df_diffex = timeseriesdf.diff().diff(365).dropna()
diff_v2ex = timeseriesdf['close'].diff().diff(365).dropna()
time_seriesex = timeseriesdf.index
time_seriesex = time_seriesex.to_frame(index = True)
diff_v2ex.plot()

result=adfuller (diff_v2ex)
print('Test Statistic: %f' %result[0])
print('p-value: %f' %result[1])
print('Critical values:')
for key, value in result[4].items ():
     print('\t%s: %.3f' %(key, value))

result_kpss_ct_log=kpss(diff_v2ex,regression="ct")
print('Test Statistic: %f' % np.round(result_kpss_ct_log[0],2))
print('p-value: %f' %result_kpss_ct_log[1])
print('Critical values:')
for key, value in result_kpss_ct_log[3].items():
     print('\t%s: %.3f' %(key, value))

plt.figure(figsize=(20,8))
plt.plot(diff_v2ex,label="after")
plt.plot(timeseriesdf,label="before")
plt.tick_params(
    axis='x',        
    which='both',   
    bottom=False,      
    top=False,        
    labelbottom=False)
plt.legend()
plt.show()


stationarydf2 = diff_v2ex

# Let's use original and both these stationary datasets for Auto-ARIMA

test_size = round((timeseriesdf.shape[0]) - 730)

Train = timeseriesdf[:test_size]
Test = timeseriesdf[test_size:]

# Auto ARIMA on complete Dataset
import itertools
from math import sqrt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#Standard ARIMA Model
ARIMA_model = pm.auto_arima(data1['close'], 
                      start_p=1, 
                      start_q=1,
                      test='adf', # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                      d=None,# let model determine 'd'
                      seasonal=False, # No Seasonality for standard ARIMA
                      trace=False, #logs 
                      error_action='warn', #shows errors ('ignore' silences these)
                      suppress_warnings=True,
                      stepwise=True)

ARIMA_model.plot_diagnostics(figsize=(20,12))
plt.show()

from pandas.tseries.frequencies import DAYS
def forecast(ARIMA_model, periods=730):
    # Forecast
    n_periods = periods
    fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(data.index[-1] + pd.DateOffset(days=1), periods = n_periods, freq='D')
    
    # make series for plotting purpose
    fitted_series = pd.Series(fitted.values, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(data["close"])
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=.15)

    plt.title("ARIMA - Forecast of Close Price")
    plt.show()

arimaplot0 = forecast(ARIMA_model)
#########################################################################
# Naive Predictions / Persistence / Base model

# evaluate a persistence model
print('Dataset %d, Validation %d' % (len(Train), len(Test)))

train = Train.close.values.astype('float32')

test = Test.close.values.astype('float32')

# walk-forward validation
history = [x for x in train]


predictions = list()

for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
    
    # observation
    obs = test[i]
    history.append(obs)    
    #print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))

# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
###########################################################################
#LSTM ANN

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

new_data=data.drop(['symbol','open','high','low','volume','date','returns percentage'],axis=1)


#creating train and test sets
dataset = new_data
test_size = round(0.25 * (dataset.shape[0]+1))

train = dataset[:-test_size]
valid = dataset[-test_size:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(46,len(train)):
    x_train.append(scaled_data[i-46:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 896 values, using past 46 from the train data
inputs = new_data[len(new_data) - len(valid) - 46:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(46,inputs.shape[0]):
    X_test.append(inputs[i-46:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

# Results
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))




#for plotting
lstmplot0 = plt.figure(figsize=(25,10))
train = dataset[:-test_size]
valid = dataset[-test_size:]
valid['Predictions'] = closing_price
plt.plot(dataset['close'], label='original')
plt.plot(valid['Predictions'],label='predicted')
plt.legend()
##########################################################################
#FB PROPHET

import fbprophet
from fbprophet import Prophet

data2 = data
data2['ds'] = pd.to_datetime(data['date'])
data2['y'] = (data2['close'])
data2 = data2[['ds','y']].reset_index(drop = True)
data2.info()

model = Prophet()
model.fit(data2)


future = model.make_future_dataframe(periods = 730)


pred = model.predict(future)

pred.yhat[pred.yhat < 0] = 0
pred.yhat_lower[pred.yhat_lower < 0] = 0
pred.yhat_upper[pred.yhat_upper < 0] = 0
pred.trend_upper[pred.trend_upper < 0] = 0
pred.trend_lower[pred.trend_lower < 0] = 0


fbplot0 = model.plot(pred)

fbplot1 = model.plot_components(pred)

se = np.square(pred.loc[:, 'yhat'] - data2.y)
mse = np.mean(se)
rmse = np.sqrt(mse)
'''


##########################################################################

if MODEL == 'Model Based':
	model(COMPANY)
	
if MODEL == 'Data Driven':
	datad(COMPANY)

if MODEL == 'ARIMA':
	arima(COMPANY)
'''


if MODEL == 'LSTM Artificial Neural Network':
	st.header('LSTM Artificial Neural Network based Forecasting')
	st.subheader('Forecast by LSTM ANN')
	st.write(lstmplot0)
	

if MODEL == 'FB Prophet':
	st.header('Forecast by FB Prophet Model')
	st.subheader('Predicted Result')
	st.write(fbplot0)
	st.subheader('Other Components of FBPROPHET')
	st.write(fbplot1)
'''
