# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:49:58 2021

@author: johnpsom
"""


import warnings
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from stocks import stocks_list, download_from_yahoo
from portfolio_functions import momentum_score, get_latest_prices
from portfolio_functions import backtest_portfolio, get_portfolio
from portfolio_functions import rebalance_portfolio, backtest_portfolio2


def load_data(tickers_sp500, start, end, interval='1d'):
    return download_from_yahoo(tickers_sp500, start, end, '1d')


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")
# get current date as end_date
end_date = datetime.strftime(datetime.now().date(), '%Y-%m-%d')
# get as start date 1200 days ago
start_date = datetime.strftime(
    datetime.now() - timedelta(days=1500), '%Y-%m-%d')
# Load rows of data into a dataframe.
stocks = stocks_list()
stocks_data = load_data(stocks, start_date, end_date, '1d')
# create the closing prices dataframe
l_close = pd.DataFrame(columns=['stock', 'date', 'last_price', 'len_prices'])

i = 1
for ticker in stocks:
    last_close = stocks_data['Adj Close'].iloc[-1][ticker]
    last_date = end_date
    len_values = len(stocks_data)
    l_close = l_close.append({'stock': ticker, 'date': last_date, 'lastprice': last_close,
                              'len_prices': len_values}, ignore_index=True)

close_data = stocks_data['Adj Close']
close_data.dropna(how='all', axis=1, inplace=True)
l_close_min = l_close['len_prices'].min()


# Declare Constants
port_value = 50000
new_port_value = 0
df = close_data
# momentum_window = v
# minimum_momentum = 70
# portfolio_size=15
# cutoff=0.1
# how much cash to add each trading period
# tr_period=20 #trading period, 20 is a month,10 in a fortnight, 5 is a week, 1 is everyday
# dataset=800 #start for length of days used for the optimising dataset
# l_days=600  #how many days to use in optimisations
backtest_results = pd.DataFrame(columns=['trades', 'momentum_window',
                                         'minimum_momentum', 'portfolio_size',
                                         'tr_period', 'cutoff', 'tot_contribution',
                                         'final port_value',
                                         'cumprod', 'tot_ret', 'drawdown'])

# backtest on the 90% of our dataset
x = int(0.9*l_close_min)
# .head(x)  # backtest dataframe of first x values from total prices
df_bt = df
# validate dataframe of the rest prices like a forward test
df_vld = df.tail(x)

for backtest_days in [200]:
    # it doesn't matter if you add more days in the backtest since we re looking for a set of
    # parameters that give a max positive return during the backtest and want to retain this behaviour
    # for a little longer and equally to the trading period.
    for momentum_window in range(90, 510, 30):
        for minimum_momentum in range(70, 190, 30):
            for portfolio_size in range(5, 25, 5):
                for trading_period in [5, 10, 20]:
                    for cutoff in [1, 5, 10, 15]:
                        cutoff /= 100
                        portfolio_value = 50000
                        backtest_dataset = len(df_bt)-backtest_days
                        lookback_days = momentum_window
                        added_value = trading_period*0
                        bt_result, plotted_portval = backtest_portfolio(df_bt, backtest_dataset,
                                                                        lookback_days, momentum_window,
                                                                        minimum_momentum, portfolio_size,
                                                                        trading_period, cutoff,
                                                                        portfolio_value, added_value)
                        backtest_results = backtest_results.append(
                            bt_result, ignore_index=True)
                        print(bt_result)
                        plt.plot(plotted_portval['portvalue'])
                        print(backtest_results.sort_values(
                            by=['tot_ret']).tail(2))

print(backtest_results.sort_values(by=['tot_ret']).tail(10))
best_backtest_result = backtest_results.sort_values(
    by=['tot_ret']).tail(1).reset_index(drop=True)
print(best_backtest_result)

# validate the best of the best portfolio
portfolio_value = 50000
momentum_window = int(best_backtest_result.loc[0, 'momentum_window'])
minimum_momentum = int(best_backtest_result.loc[0, 'minimum_momentum'])
portfolio_size = int(best_backtest_result.loc[0, 'portfolio_size'])
cutoff = (best_backtest_result.loc[0, 'cutoff'])
trading_period = int(best_backtest_result.loc[0, 'tr_period'])
validation_dataset = len(df_vld)-backtest_days
lookback_days = momentum_window
added_value = 0
bt_result = backtest_portfolio2(df_vld, validation_dataset,
                                lookback_days, momentum_window,
                                minimum_momentum, portfolio_size,
                                trading_period, cutoff,
                                portfolio_value, added_value)
print(bt_result)


# create final and new portfolio
df_old = pd.read_csv('old_portfolio.csv').iloc[:, 1:]
cash = df_old.loc[df_old['stock'] == 'CASH']['value'].values[0]
latest_prices = get_latest_prices(df)
new_price = []
for symbol in df_old['stock'][:-1].values.tolist():
    new_price.append(latest_prices[symbol])
new_price.append(cash)
df_old['new_prices'] = new_price
df_old['new_value'] = df_old['new_prices']*df_old['shares']
portfolio_value = 0.9*(cash+df_old['new_value'].sum())
# portfolio_value=2200
df_tr = df.tail(momentum_window)
df_m = pd.DataFrame()
m_s = []
st = []
for s in stocks:
    st.append(s)
    m_s.append(momentum_score(df_tr[s].tail(momentum_window)))
df_m['stock'] = st
df_m['momentum'] = m_s
dev = df_m['momentum'].std()
# Get the top momentum stocks for the period
df_m = df_m.sort_values(by='momentum', ascending=False)
df_m = df_m[(df_m['momentum'] > minimum_momentum-0.5*dev) &
            (df_m['momentum'] < minimum_momentum+1.9*dev)].head(portfolio_size)
# Set the universe to the top momentum stocks for the period
universe = df_m['stock'].tolist()
print(universe)
# Create a df with just the stocks from the universe
df_buy = get_portfolio(universe, df_tr, portfolio_value, cutoff, df_m)[0]
df_buy = df_buy.reset_index()
old_cash = df_buy.loc[df_buy.stock == 'CASH', 'value'].values[0]
new_cash = old_cash+(df_old['new_value'].sum()-df_buy['value'].sum())
df_buy.loc[df_buy.stock == 'CASH', 'value'] = new_cash
# here you should add the diff of cash to df_buy CASH and then save portfolio
print(df_old)
print(' ')
print(df_buy)
print(df_old['new_value'].sum(), df_buy['value'].sum())
print('To rebalance your old portfolio to the new you should')
# rebalance with old portfolio
rebalance_portfolio(df_old, df_buy)
