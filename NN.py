import yfinance as yf
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots


warnings.simplefilter(action='ignore', category=FutureWarning)

lr = LinearRegression()


def data_get(ticker1, start1, end1):
    df1 = yf.download(ticker1, start=start1, end=end1, interval='1d')
    return df1


def data_process(df, column, split):
    percent_split = split
    row = int(len(np.array(df)) * percent_split)
    dataX = df.iloc[:row]
    dataY = df.iloc[row:]

    X = dataX.iloc[1:]
    Y = dataX.iloc[:-1]
    y = Y[[column]]

    X_test = dataY.iloc[1:]
    Y_test = dataY.iloc[:-1]
    y_test = Y_test[[column]]

    #print(len(np.array(X_test)))

    return X, y, X_test, y_test


def neural_network(ticker, start, end, column, split):
    global lr
    df = data_get(ticker, start, end)

    CustomStrategy = ta.Strategy(
        name="Momo and Volatility",
        description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
        ta=[
            #{"kind": "sma", "length": 50},
            #{"kind": "sma", "length": 200},
            #{"kind": "bbands", "length": 20},
            {"kind": "rsi"},
            {"kind": "stoch"},
            {"kind": "bbands"},
            {"kind": "ema"},
            {"kind": "ao"},
            {"kind": "apo"},
            {"kind": "brar"},
            {"kind": "cci"},
            {"kind": "ha"},
            {"kind": "aberration"},
            {"kind": "accbands"},
            {"kind": "adx"},
            {"kind": "bias"},
            {"kind": "cmf"},
            {"kind": "midprice"},
            {"kind": "pvol"},
            #{"kind": "macd", "fast": 8, "slow": 21},
            #{"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
        ]
    )


    df.ta.strategy(CustomStrategy) #error here
    #print(df)
    df = df.dropna()
    #print(df)


    #print("__________________________________")

    #print(df)

    X, y, X_test, y_test = data_process(df, column, split)

    lr.fit(np.array(X), np.array(y))

    a = lr.predict(np.array(X_test))
    b = np.array(y_test)

    t = 0

    # Accuracy Calculator
    for z in range(len(a)):
        w = a[z]
        v = b[z]

        if z != 0:
            u = b[z - 1]
            if (w[0] < u[0] and v[0] < u[0]) or (w[0] > u[0] and v[0] > u[0]):
                t += 1

    # Output Accuracy

    print(lr.score(np.array(X_test), np.array(y_test)))
    print("High Direction Accuracy: " + str(t / len(a) * 100) + "%")
    #print(df)
    return df, X, y, X_test, y_test


def graph(df,X, y, X_test, y_test,ticker ):

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        )

    fig.add_trace(
        go.Candlestick(x=df.axes[0].tolist(), open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                       increasing_line_color='rgb(0,255,0)', decreasing_line_color='rgb(255,0,0)'))

    z1 = lr.predict(X_test)
    z2 = X_test.axes[0].tolist()
    z3 = np.array(y_test)

    #print(z1)
    #print(z3)
    print(z2)
    for k in range(len(z1)-1):

        w=z1[k]
        v=z3[k]

        if k != 0:
            u = z3[k - 1]
            print(w[0],u[0],v[0])
            if (w[0] < u[0] and v[0] < u[0]):
                fig.add_shape(type="line", x0=str(z2[k]).split()[0], x1=str(z2[k]).split()[0], y0=0.2, y1=0,
                              line=dict(color="rgb(255,0,0)", width=2))
            elif (w[0] > u[0] and v[0] > u[0]):
                fig.add_shape(type="line", x0=str(z2[k]).split()[0], x1=str(z2[k]).split()[0], y0=0.2, y1=0,
                              line=dict(color="rgb(0,255,0)", width=2))


        '''
        if z3[k][0]-z3[k+1][0] <= 0 and z3[k][0]-z1[k]<=0: #if positive
            fig.add_shape(type="line", x0=str(z2[k]).split()[0], x1=str(z2[k]).split()[0], y0=0.2, y1=0,
                          line=dict(color="rgb(0,255,0)", width=2))
        else:
            fig.add_shape(type="line", x0=str(z2[k]).split()[0], x1=str(z2[k]).split()[0], y0=0.2, y1=0,
                          line=dict(color="rgb(255,0,0)", width=2))
        '''



        '''
        if z1[k][0] >= z3[k][0]:
            fig.add_shape(type="line", x0=str(z2[k]).split()[0], x1=str(z2[k]).split()[0], y0=0.2, y1=0,
                          line=dict(color="rgb(0,255,0)", width=2))
        else:
            fig.add_shape(type="line", x0=str(z2[k]).split()[0], x1=str(z2[k]).split()[0], y0=0.2, y1=0,
                          line=dict(color="rgb(255,0,0)", width=2))
        '''

    fig.update_layout(
        title=ticker,
        yaxis_title='Price',
        xaxis_title='Date',
        font=dict(
            family='Courier New, monospace',
            size=10,
            color='#7f7f7f'
        ),
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        ),
        height=700,
        paper_bgcolor='rgb(0,0,0)',
        plot_bgcolor='rgb(0,0,0)'
    )

    fig.update_shapes(dict(xref='x', yref='paper'))
    fig.update_yaxes(automargin=True)

    import dash
    import dash_core_components as dcc
    import dash_html_components as html

    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])

    print("RUN SERVER")
    fig.show()
    #return app.run_server(port=5000,debug=True, use_reloader=False)

    #return fig.show()



def run():
    #if __name__ == '__main__':
    try:
        ticker = input("Ticker: ")

        vc,X, y, X_test, y_test = neural_network(ticker, "2017-01-01", "2020-11-26", "close", 0.95)
        #return graph(vc,X, y, X_test, y_test,ticker )
        graph(vc, X, y, X_test, y_test, ticker)
    except ValueError:
        pass
    #return "hi"





