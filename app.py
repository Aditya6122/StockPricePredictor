import math
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from io import StringIO
import pandas as pd
import plotly.express as px
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
from stqdm import stqdm
from plotly import tools
import plotly.graph_objs as go

st.set_page_config(layout="wide")


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out


def prepare_data(data,investment_type):
    if(investment_type == 'intraday'):
        data.drop(data.columns[1],axis=1,inplace=True)
        data.dropna(inplace=True)
    else:
        data = data[['Date ','close ']]
        data = data[::-1]

    data.reset_index(inplace=True)
    data.drop('index',axis=1,inplace=True)
    data.rename(columns={data.columns[0]:'date',data.columns[1]:'close'},inplace=True)

    return data

def split_data(stock, lookback):
    data_raw = stock.to_numpy()
    data = []

    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    test_set_size = int(np.round(0.3*data.shape[0]))
    train_set_size = data.shape[0] - test_set_size
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    x_test = data[train_set_size:,:-1,:]
    y_test = data[train_set_size:,-1,:]

    return [x_train, y_train, x_test, y_test]

def min_max_scaler(prices):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return [scaler.fit_transform(prices.values.reshape(-1,1)),scaler]

def data_to_tensor(x_train,y_train,x_test,y_test):
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    return [x_train,y_train,x_test,y_test]

def inverse_scaler(scaler,y_train_pred,y_train_gru,y_test_pred,y_test_gru):
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_gru.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_gru.detach().numpy())
    return [y_train_pred,y_train,y_test_pred,y_test]

st.title('Stock Price Predictor')

uploaded_file = st.file_uploader("Upload Stocks Dataset File Here (NSE format)")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    stringio = StringIO(bytes_data.decode("utf-8"))
    string_data = stringio.read()
    dataframe = pd.read_csv(uploaded_file)
    data = prepare_data(dataframe,'intraday')
    fig = px.line(data, x="date", y="close", title='Intraday Stock Data')
    st.plotly_chart(fig,use_container_width=True)

    price  = data[['close']]
    price['close'],scaler = min_max_scaler(price['close'])
    lookback = 20
    x_train, y_train, x_test, y_test = split_data(price,lookback)
    x_train, y_train_gru, x_test, y_test_gru = data_to_tensor(x_train, y_train, x_test, y_test)

    input_dim = 1
    hidden_dim = 32
    num_layers = 2
    output_dim = 1
    num_epochs = 100

    model = GRU(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    hist = np.zeros(num_epochs)
    
    with st.spinner('Please wait....Training the intraday sequence'):
        for t in stqdm(range(num_epochs)):
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred, y_train_gru)
            print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    st.success('!! Done with the training and identifying patterns')

    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_gru.detach().numpy()))

    trace1 = go.Scatter(x = predict.index,y=predict[0],name='predicted')
    trace2 = go.Scatter(x = original.index,y=original[0],name='original')
    trace3 = go.Scatter(x = np.arange(0,len(hist)-1),y=hist,name='Loss values')

    fig1 = tools.make_subplots(rows = 1, cols = 2,subplot_titles=('Original Vs. Predicted',  'Loss Values'))
    fig1.append_trace(trace1, 1, 1)
    fig1.append_trace(trace2, 1, 1)
    fig1.append_trace(trace3, 1, 2)

    fig1['layout'].update(height = 600, width = 800)
    fig1['layout']['xaxis']['title']='Time'
    fig1['layout']['xaxis2']['title']='Iterations'
    fig1['layout']['yaxis']['title']='Prices'
    fig1['layout']['yaxis2']['title']='Loss'

    st.plotly_chart(fig1,use_container_width=True)

    y_test_pred = model(x_test)
    y_train_pred,y_train,y_test_pred,y_test = inverse_scaler(scaler,y_train_pred,y_train_gru,y_test_pred,y_test_gru)
    
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    acc_test_score = round(r2_score(y_test,y_test_pred)*100,2)
    acc_train_score = round(r2_score(y_train,y_train_pred)*100,2)

    train_score, test_score = st.columns(2)
    train_score.metric("R2_score: Train data",str(acc_train_score))
    test_score.metric("R2_score: Test data",str(acc_test_score))

    trainPredictPlot = np.empty_like(price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

    testPredictPlot = np.empty_like(price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

    original = scaler.inverse_transform(price.values.reshape(-1,1))

    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)

         
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                        mode='lines',
                        name='Train prediction')))
    fig2.add_trace(go.Scatter(x=result.index, y=result[1],
                        mode='lines',
                        name='Test prediction'))
    fig2.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                        mode='lines',
                        name='Actual Value')))

    fig2['layout'].update(title_text='Final Predictions')
    fig2['layout']['xaxis']['title']='Time'
    fig2['layout']['yaxis']['title']='Prices'

    st.plotly_chart(fig2,use_container_width=True)
    
    