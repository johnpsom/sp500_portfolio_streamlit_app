# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:00:54 2021

@author: johnpsom

this is a streamlit app to create portfolios
    from the sp500 stocks universe

"""


# import local functions and variables used
from datetime import datetime
from datetime import timedelta
from enum import Enum
from io import BytesIO, StringIO
from typing import Union
import streamlit as st
import pandas as pd

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.risk_models import CovarianceShrinkage

from stocks import stocks_list, download_from_yahoo
from portfolio_functions import momentum_score, capm_return, get_latest_prices
from portfolio_functions import select_columns, download_button
from portfolio_functions import rebalance_portfolio, backtest_portfolio


STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

FILE_TYPES = ["csv"]


class FileType(Enum):
    """Used to distinguish between file types"""
    CSV = "csv"


def get_file_type(file: Union[BytesIO, StringIO]) -> FileType:
    """The file uploader widget does not provide information on the type of file uploaded so we have
    to guess using rules or ML
    I've implemented rules for now :-)
    Arguments:
        file {Union[BytesIO, StringIO]} -- The file uploaded
    Returns:
        FileType -- A best guess of the file type
    """
    return FileType.CSV


@st.cache(ttl=24 * 60 * 60)
def load_data(tickers_sp500, start, end, interval='1d'):
    return download_from_yahoo(tickers_sp500, start, end, '1d')


st.set_page_config(layout="wide")
st.markdown(
    '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" '
    'integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
    unsafe_allow_html=True)
st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #CE2B08;">
  <a class="navbar-brand" target="_blank">getyour.Portfolio@gmail.com</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a href="https://share.streamlit.io/johnpsom/sp500_portfolio_streamlit_app/main/main.py" target="_blank">HomePage</a>
      </li>
      <li class="nav-item active">
        <a href="https://medium.com/@ioannis.psomiadis/a-streamlit-app-to-create-and-test-portfolios-of-sp500-stocks-956a7b79275" target="_blank">Article on Medium</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

st.markdown('''
#                       **Portfolio Optimization and Creation of SP500 stocks**
#### **Beware** that this app is created for educational and informational purposes only. 
#### Stock markets are volatile and there is risk involved and also loss of money.
#### The creator of this app is in no case responsible of what you do with your money. 
#### Comments are welcome and accepted in the article's page on [medium.com](https://medium.com/@ioannis.psomiadis/a-streamlit-app-to-create-and-test-portfolios-of-sp500-stocks-956a7b79275) or email me at <getyour.portfolio@gmail.com>.
#### Check article on [medium.com](https://medium.com/@ioannis.psomiadis/a-streamlit-app-to-create-and-test-portfolios-of-sp500-stocks-956a7b79275) for instructions and guidance on what this app does and how it works.

#### Choose values for the parameters in the sidebar accordingly and check out the portfolio
#### given as result.
''')

# get current date as end_date
end_date = datetime.strftime(datetime.now().date(), '%Y-%m-%d')
# get as start date 1500 days ago
start_date = datetime.strftime(datetime.now() - timedelta(days=1500), '%Y-%m-%d')
# Load rows of data into a dataframe.
stocks = stocks_list()
stocks_data = load_data(stocks, start_date, end_date, '1d')
# create the closing prices dataframe
l_close = pd.DataFrame(columns=['stock', 'date', 'last_price', 'len_prices'])
close_data = stocks_data['Adj Close']
#i = 1
for ticker in stocks:
    last_close = stocks_data['Adj Close'].iloc[-1][ticker]
    last_date = end_date
    len_values = len(stocks_data)
    l_close = l_close.append({'stock': ticker, 'date': last_date, 'lastprice': last_close,
                              'len_prices': len_values}, ignore_index=True)
    #df_temp = stocks_data['Adj Close'].loc[:, [ticker]].rename(
    #    columns={'Adj Close': ticker})
    #if i == 1:
        #close_data = df_temp
    #    i = i + 1
    #else:
        #close_data = close_data.merge(df_temp, how='inner', on='Date')

#close_data = close_data.copy()
#close_data.dropna(how='all', axis=1, inplace=True)
l_close_min = l_close['len_prices'].min()

best_res = pd.DataFrame(columns=['trades', 'momentum_window', 'minimum_momentum', 'portfolio_size',
                                 'tr_period', 'cutoff', 'tot_contribution', 'final port_value',
                                 'cum_prod', 'tot_ret', 'draw_down'])
df = close_data
df_tr = df

# -----Γενικές παράμετροι
st.sidebar.write('Portfolio Parameters')
port_value = st.sidebar.slider(
    'Initial Capital to invest $', 10000, 100000, 50000, 5000)
momentum_window = st.sidebar.slider(
    'Number of days to use to calculate our momentum indicator.', 90, 500, 120, 10)
minimum_momentum = st.sidebar.slider(
    'Minimum value of the momentum indicator so that a stock can be included in our portfolio.', 70, 180, 120, 10)
portfolio_size = st.sidebar.slider(
    'Maximum Portfolio Size No of stocks.', 5, 50, 5, 1)
cutoff = st.sidebar.slider(
    'Minimum weight of a stock in our portfolio.', 0.01, 0.20, 0.10, 0.01)
added_value = st.sidebar.slider(
    'How much money are you going to add to your investment $/day. ', 0, 50, 0, 1)
history_bt = st.sidebar.slider(
    'Days of backtesting our Portfolio (Υ days).', 100, 400, 100, 100)
df_m = pd.DataFrame()
m_s = []
sto = []
for s in stocks:
    sto.append(s)
    m_s.append(momentum_score(df_tr[s].tail(momentum_window)))
df_m['stock'] = sto
df_m['momentum'] = m_s
dev = df_m['momentum'].std()
# Get the top momentum stocks for the period
df_m = df_m.sort_values(by='momentum', ascending=False)
df_m = df_m[(df_m['momentum'] > minimum_momentum - 0.5 * dev) & (df_m['momentum'] < minimum_momentum + 1.9 * dev)].head(
    portfolio_size)
# Set the universe to the top momentum stocks for the period
universe = df_m['stock'].tolist()
# Create a df with just the stocks from the universe
df_t = select_columns(df_tr, universe)
st.write('Latest prices of our selected stocks.')
st.write(df_t.tail())
# Calculate portfolio mu and S
mu = capm_return(df_t)
S = CovarianceShrinkage(df_t).ledoit_wolf()
# Optimise the portfolio for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.min_volatility()
cleaned_weights = ef.clean_weights(cutoff=cutoff, rounding=3)
ef.portfolio_performance()

st.subheader('Proposed Optimized Portfolio')
st.write('The proposed Portfolio below has the following performance.')
st.write('Initial Value of : ' + str(port_value) + '$')
st.write('Sharpe Ratio  : ' + str(round(ef.portfolio_performance()[2], 2)))
st.write('Annual return : ' +
         str(round(ef.portfolio_performance()[0] * 100, 2)) + '%')
st.write('Volatility    : ' +
         str(round(ef.portfolio_performance()[1] * 100, 2)) + '%')
# Allocate
latest_prices = get_latest_prices(df_t)
da = DiscreteAllocation(cleaned_weights,
                        latest_prices,
                        total_portfolio_value=port_value
                        )
allocation = da.greedy_portfolio()[0]
non_trading_cash = da.greedy_portfolio()[1]

# Put the stocks and the number of shares from the portfolio into a df
symbol_list = []
mom = []
w = []
num_shares_list = []
l_price = []
tot_cash = []
for symbol, num_shares in allocation.items():
    symbol_list.append(symbol)
    w.append(cleaned_weights[symbol])
    num_shares_list.append(num_shares)
    l_price.append(latest_prices[symbol])
    tot_cash.append(num_shares * latest_prices[symbol])

df_buy = pd.DataFrame()
df_buy['stock'] = symbol_list
df_buy['weights'] = w
df_buy['shares'] = num_shares_list
df_buy['price'] = l_price
df_buy['value'] = tot_cash
st.write(
    f'Invested in stock {round(df_buy["value"].sum(),2)}$ or {round(100 * df_buy["value"].sum() / port_value,2)}% of our portfolio')
st.write(
    f'Non invested Cash :{round(port_value - df_buy["value"].sum(),2)}€ ή το {round(100 - 100 * df_buy["value"].sum() / port_value,2)}% of our portfolio.')

df_buy = df_buy.append({'stock': 'CASH',
                        'weights': round(1 - df_buy['value'].sum() / port_value, 2),
                        'shares': 1,
                        'price': round(port_value - df_buy['value'].sum(), 2),
                        'value': round(port_value - df_buy['value'].sum(), 2)}, ignore_index=True)

st.dataframe(df_buy)
st.write('On the above dataframe we see in the "weights" column the percentage of each stock in our portfolio.')
st.write('In the "shares" columns the number of shares we allocate, at what "price" be bought them and ')
st.write('the total "value" of money allocated on each stock')

st.write(
    'If you want to keep this portfolio you can download it as a csv file by ging a name and pressing the Save button')
file_name = st.text_input(
    'Name your portfolio and press the button below to save it', value="My Portfolio", key=1)
if st.button('Save this Portfolio', key=1):
    file_name = file_name + '.csv'
    download_button_str = download_button(
        df_buy, file_name, f'Click here to download {file_name}', pickle_it=False)
    st.markdown(download_button_str, unsafe_allow_html=True)

st.subheader(
    'If you have previoulsy used this App and you have downloaded a Portfolio upload it to see its performance today.')
st.markdown(STYLE, unsafe_allow_html=True)
file = st.file_uploader(
    "Drag and drop hear your Portfolio (*.csv)", type='csv')
show_file = st.empty()
if not file:
    show_file.info("")
else:
    df_old = pd.read_csv(file)
    df_old = df_old.rename(columns={'price': 'bought price'})
    last_price = []
    new_values = []
    new_weights = []
    pct = []
    for stock in list(df_old.iloc[:-1]['stock']):
        last_price.append(df.iloc[-1][stock])
        nv = df_old.loc[df_old['stock'] == stock,
                        'shares'].values[0] * df.iloc[-1][stock]
        new_values.append(nv)
        pt = round(100 * (df.iloc[-1][stock] / df_old.loc[df_old['stock']
                   == stock, 'bought price'].values[0] - 1), 2)
        pct.append(pt)
    last_price.append(0)
    pct.append(0)
    df_old['last price'] = last_price
    new_values.append(df_old.iloc[-1]['value'])
    df_old['new value'] = new_values
    df_old['pct_change%'] = pct
    new_port_value = df_old['new value'].sum()
    for stock in list(df_old.iloc[:-1]['stock']):
        new_weights.append(
            df_old.loc[df_old['stock'] == stock, 'shares'].values[0] * df.iloc[-1][stock] / new_port_value)
    new_weights.append(df_old.iloc[-1]['new value'] / new_port_value)
    df_old['new weights'] = new_weights
    st.write(f'Portfolio initial value was :{df_old["value"].sum()}$')
    st.write(f'And now it is : {round(new_port_value, 2)} €')
    st.write(
        f'with a return of {100 * round(new_port_value / df_old["value"].sum() - 1, 2)}%')
    st.dataframe(df_old)
    file.close()
    rebalance_portfolio(df_old, df_buy)

st.markdown('''**Below you see a backtest for the portfolio with the chosen parameters if we rebalanced it every week (5 days),
               fortnight (10 days) and month (20 days) in the last Y days you chose in the sidebar.**''')
bt_days = l_close_min - history_bt

rs5 = backtest_portfolio(df, dataset=bt_days, l_days=bt_days - momentum_window, momentum_window=momentum_window,
                         minimum_momentum=minimum_momentum, portfolio_size=portfolio_size, tr_period=5, cutoff=cutoff,
                         port_value=port_value, a_v=added_value)[0]

chart_data5 = backtest_portfolio(df, dataset=bt_days, l_days=bt_days - momentum_window, momentum_window=momentum_window,
                                 minimum_momentum=minimum_momentum, portfolio_size=portfolio_size, tr_period=5, cutoff=cutoff,
                                 port_value=port_value, a_v=added_value)[1]

st.write(
    f'With an initial investment of "{port_value} dollars", we would have rebalanced {rs5["trades"]} times, every 5 days, and would have a return of {round(rs5["tot_ret"], 2)} % and accumulated {round(rs5["final port_value"], 2)}$')
with st.expander("See a bar plot of the portfolio value change in time"):
    st.write("""
         The chart below  shows the evolution in time of our portfolio.
     """)
    st.bar_chart(data=chart_data5.loc[:, ['portvalue']], width=0, height=0, use_container_width=True)
    
rs10 = backtest_portfolio(df, dataset=bt_days, l_days=bt_days - momentum_window, momentum_window=momentum_window,
                          minimum_momentum=minimum_momentum, portfolio_size=portfolio_size, tr_period=10, cutoff=cutoff,
                          port_value=port_value, a_v=added_value)[0]
chart_data10 = backtest_portfolio(df, dataset=bt_days, l_days=bt_days - momentum_window, momentum_window=momentum_window,
                                  minimum_momentum=minimum_momentum, portfolio_size=portfolio_size, tr_period=10, cutoff=cutoff,
                                  port_value=port_value, a_v=added_value)[1]

st.write(
    f'With an initial investment of "{port_value} dollars", we would have rebalanced {rs10["trades"]} times, every 10 days, and would have a return of {round(rs10["tot_ret"], 2)}% and accumulated {round(rs10["final port_value"], 2)}$')
with st.expander("See a bar plot of the portfolio value change in time"):
    st.write("""
         The chart below shows the evolution in time of our portfolio.
     """)
    st.bar_chart(data=chart_data10.loc[:, ['portvalue']], width=0, height=0, use_container_width=True)

rs20 = backtest_portfolio(df, dataset=bt_days, l_days=bt_days - momentum_window, momentum_window=momentum_window,
                          minimum_momentum=minimum_momentum, portfolio_size=portfolio_size, tr_period=20, cutoff=cutoff,
                          port_value=port_value, a_v=added_value)[0]
chart_data20 = backtest_portfolio(df, dataset=bt_days, l_days=bt_days - momentum_window, momentum_window=momentum_window,
                                  minimum_momentum=minimum_momentum, portfolio_size=portfolio_size, tr_period=20, cutoff=cutoff,
                                  port_value=port_value, a_v=added_value)[1]

st.write(
    f'With an initial investment of "{port_value} dollars", we would have rebalanced {rs20["trades"]} times, every 20 days, and would have a return of {round(rs20["tot_ret"], 2)} % and accumulated {round(rs20["final port_value"], 2)}$')
with st.expander("See a bar plot of the portfolio value change in time"):
    st.write("""
         The chart below  shows the evolution in time of our portfolio.
     """)
    st.bar_chart(data=chart_data20.loc[:, ['portvalue']], width=0, height=0, use_container_width=True)
