import yfinance as yf
import joblib
import datetime as dt
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense


crypto_symbols = ['BTC', 'ETH', 'LTC', 'DOGE', 'SOL', 'USDT', 'USDC', 'BNB', 'XRP', 'ADA', 'DAI', 'WTRX',
                  'DOT', 'HEX', 'TRX', 'SHIB', 'LEO', 'WBTC', 'AVAX', 'YOUC', 'MATIC', 'UNI1', 'STETH', 'LTC', 'FTT']

# load data from yfinance
def load_data(symbol):
    yf_data = yf.Ticker(f'{symbol}-USD').history(start='2014-01-01', end=dt.datetime.now(), interval='1d')
    yf_data.reset_index(inplace=True)
    yf_data.drop(['Dividends', 'Stock Splits'], axis='columns', inplace=True)
    return yf_data
st.set_page_config(layout='wide')

st.sidebar.write('developer: Oluwaseun Akinkuolie')

#image1 = Image.open('My logo.jpeg')
#st.sidebar.image(image1, width=300)
selected_coins = st.sidebar.selectbox("Select Coin Interest", crypto_symbols)
#interval = int(st.sidebar.number_input('Input Interval'))
intervals = {
    '1 day': 1,
    '1 week': 7,
    '1 month': 30,
    'Quarter': 90
}

selected_interval = st.sidebar.selectbox('Select Interval measurement', list(intervals.keys()))
interval = intervals[selected_interval]


with st.spinner('Load data...'):
    data = load_data(selected_coins)

st.title('SoliGENCE  COIN TRADING SYSTEM')
st.markdown('''This app gives an estimate of the price of a given currency in the future and shows you dates you can 
purchase a coin to give you a profit or even a loss''')
st.success('Welcome to SoliGENCE, profit maximization is our goal here!')


st.subheader('Correlation of 10 most popular Coins')
st.markdown('''The correlation shows coins that are likely to have the same trend in the market. Coins having correlation
 of about 0.5 are correlated(both +ve and -ve) while those having above 0.7 have very strong correlation correlated''')
image = Image.open('download.png')
st.image(image, width=600)


def visualize_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data.Close.rolling(30).mean(), name='Moving average'))
    fig.update_layout(paper_bgcolor='lightgrey', title_text=f'Time Series and Moving Average of {selected_coins}', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

visualize_raw_data()

# Forecasting

df_train = data[['Date', 'Close']]


df_train = df_train.rename(columns={'Date':"ds", "Close": 'y'})

import io

# Open the file with the appropriate encoding using io.open()
with io.open('model2.joblib', 'rb') as f:
    joblib_model = joblib.load(f)


# joblib_model = joblib.load(open('model2.joblib'),encoding='latin1')
future = joblib_model.make_future_dataframe(periods=int(interval))
prediction = joblib_model.predict(future)



st.subheader('Visualization of Forecast data')
fig1 = plot_plotly(joblib_model, prediction)
st.plotly_chart(fig1)

st.subheader('Forecast Components')
fig2 = joblib_model.plot_components(prediction)
st.write(fig2)

# Load the trained LSTM model
model = keras.models.load_model('crypto_price_prediction_model.h5')
# model = joblib.load('model.joblib')

# Set up the list of crypto symbols
crypto_symbols = ['BTC', 'ETH', 'LTC', 'DOGE', 'SOL', 'USDT', 'USDC', 'BNB', 'XRP', 'ADA', 'DAI', 'WTRX',
                  'DOT', 'HEX', 'TRX', 'SHIB', 'LEO', 'WBTC', 'AVAX', 'YOUC', 'MATIC', 'UNI1', 'STETH', 'LTC', 'FTT']

# Define the mapping for interval selection
interval_mapping = {
    '1 day': '1d',
    '1 week': '1wk',
    '1 month': '1mo',
    'quarter': '3mo'
}


# Function to load data from yfinance
def load_data(symbol, interval):
    yf_data = yf.Ticker(f'{symbol}-USD').history(start='2014-01-01', end='2023-05-19', interval=interval)
    yf_data.reset_index(inplace=True)
    yf_data.drop(['Dividends', 'Stock Splits'], axis='columns', inplace=True)
    return yf_data


# Function to prepare the input data for prediction
def prepare_input_data(data, lookback):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X = []
    for i in range(len(scaled_data) - lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
    return np.array(X)


# Function to perform price prediction
def predict_price(symbol, interval, lookback):
    df = load_data(symbol, interval)
    data = df['Close'].values.reshape(-1, 1)
    input_data = prepare_input_data(data, lookback)
    input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], 1))
    predicted_scaled_price = model.predict(input_data)[-1][0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(data)
    predicted_price = scaler.inverse_transform([[predicted_scaled_price]])
    return predicted_price[0][0]

# Function to load correlation data
def load_correlation_data(symbols):
    data = yf.download(symbols, start='2014-01-01', end='2023-05-19')['Close']
    return data


# Function to calculate correlation with the selected coin
def calculate_correlation(coin, data):
    correlation = data.corrwith(data[coin])


    return correlation


# Streamlit app code
def main():
    st.title("Crypto Price Prediction")

    # Select the cryptocurrency and interval
    # selected_coin = st.selectbox("Select a cryptocurrency", crypto_symbols)
    # selected_interval = st.selectbox("Select an interval", list(interval_mapping.keys()))

    st.title("Crypto Correlation Analysis")

    # Select the cryptocurrency
    # selected_coins = st.selectbox("Select a cryptocurrency", crypto_symbols)

    # Load correlation data
    correlation_data = load_correlation_data(crypto_symbols)

    # Calculate correlations
    correlation = calculate_correlation(selected_coins, correlation_data)

    # Display positively correlated coins
    st.subheader("Positively Correlated Coins")
    positive_correlation = correlation.sort_values(ascending=False)[:10]
    st.write(positive_correlation)

    # Display negatively correlated coins
    st.subheader("Negatively Correlated Coins")
    negative_correlation = correlation.sort_values()[:10]
    st.write(negative_correlation)

    # Visualize correlations
    plt.figure(figsize=(10, 5))
    sns.barplot(x=positive_correlation.index, y=positive_correlation.values, palette="viridis")
    plt.title(f"Positive Correlation with {selected_coins}", fontsize=16)
    plt.xlabel("Cryptocurrency", fontsize=12)
    plt.ylabel("Correlation", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=negative_correlation.index, y=negative_correlation.values, palette="viridis")
    plt.title(f"Negative Correlation with {selected_coins}", fontsize=16)
    plt.xlabel("Cryptocurrency", fontsize=12)
    plt.ylabel("Correlation", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(plt)
    # Map the selected interval to the corresponding value
    interval = interval_mapping[selected_interval]

    # Get the Close price input
    # close_price = st.number_input("Enter the Close price", value=0.0)

    # Perform prediction and display the result

    todays_price = round(data['Close'].iloc[-1],2)
    if st.sidebar.button("Predict"):
        predicted_price = predict_price(selected_coins, interval, 30)
        st.write(f"Today's price of {selected_coins} is {todays_price}$")
        st.write(
            f"The predicted price of {selected_coins} in {selected_interval} will be approximately {predicted_price:.2f}$")
        value = round((predicted_price - todays_price),2)
        if value > 0:
            st.write(f"You are expected to make a profit of {value}$")
        else:
            st.write(f"You are expected to make a loss of {value}$")


    # if st.sidebar.button("Predict"):
    #     predicted_price = predict_price(selected_coin, interval, 30, close_price)
    #     st.write(
    #         f"The predicted price of {selected_coin} in {selected_interval}  will be approximately {predicted_price:.2f}")


if __name__ == "__main__":
    main()
