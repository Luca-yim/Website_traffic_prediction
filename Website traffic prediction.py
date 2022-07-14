import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

#Reading data set
data = pd.read_csv(r"C:\Users\guest\Thecleverprogrammer.csv")
#print(data.head())

#Changing date field to dd/mm/yy format
data["Date"] = pd.to_datetime(data["Date"], format = "%d/%m/%Y")

#print(data.info())

#Plot of daily traffic
#plt.style.use('fivethirtyeight')
#plt.figure(figsize=(15, 10))
#plt.plot(data["Date"], data["Views"])
#plt.title("Daily Traffic of Thecleverprogrammer.com")
#plt.show()

#Determine if the dataset is seasonal
# result = seasonal_decompose(data["Views"], model='multiplicative', period=30 )
# fig = plt.figure()
# fig = result.plot()
# fig.set_size_inches(15, 10)
# plt.show()

#finding p and q values for SARIMA model
# pd.plotting.autocorrelation_plot(data["Views"])
# plot_pacf(data["Views"], lags = 100)
# plt.show()

#Training SARIMA model
p, d, q = 5, 1, 2
model = sm.tsa.statespace.SARIMAX(data["Views"], order=(p, d, q), seasonal_order=(p, d, q, 12))
model=model.fit()
print(model.summary())

#Predictions for the next 50 days
predictions = model.predict(len(data), len(data)+50)
print(predictions)
