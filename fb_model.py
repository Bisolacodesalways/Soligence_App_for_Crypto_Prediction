import yfinance as yf
import joblib
import pandas as pd
import datetime as dt


import pandas as pd
df_train = pd.read_csv("data.csv")
# Convert the 'Date' column to datetime type
df_train['Date'] = pd.to_datetime(df_train['Date'])



df_train = df_train.rename(columns={'Date':"ds", "Close": 'y'})

from prophet import Prophet
fb = Prophet(interval_width=0.95, n_changepoints=7)
fb.fit(df_train)

# save the model
filename = "model2.joblib"
joblib.dump(fb, filename)


