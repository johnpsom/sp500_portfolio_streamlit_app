# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:49:58 2021

@author: johnpsom
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:42:37 2021

@author: IOANNIS PSOMIADIS
"""
#import streamlit

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pypfopt.expected_returns import *
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.risk_models import CovarianceShrinkage
from stocks import stocks_list, download_from_yahoo
from portfolio_functions import momentum_score, capm_return, get_latest_prices
from portfolio_functions import select_columns, download_button
from portfolio_functions import rebalance_portfolio, backtest_portfolio
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
close_data = pd.DataFrame()
i = 1
for ticker in stocks:
    last_close = stocks_data['Adj Close'].iloc[-1][ticker]
    last_date = end_date
    len_values = len(stocks_data)
    l_close = l_close.append({'stock': ticker, 'date': last_date, 'lastprice': last_close,
                              'len_prices': len_values}, ignore_index=True)
    df_temp = stocks_data['Adj Close'].loc[:, [ticker]].rename(
        columns={'Adj Close': ticker})
    if i == 1:
        close_data = df_temp
        i = i + 1
    else:
        close_data = close_data.merge(df_temp, how='inner', on='Date')

close_data = close_data.copy()
close_data.dropna(how='all', axis=1, inplace=True)
l_close_min = l_close['len_prices'].min()


# Declare Constants
port_value = 10000
new_port_value = 0
df = close_data
#momentum_window = v
#minimum_momentum = 70
# portfolio_size=15
# cutoff=0.1
# how much cash to add each trading period
# tr_period=21 #trading period, 21 is a month,10 in a fortnite, 5 is a week, 1 is everyday
# dataset=800 #start for length of days used for the optimising dataset
# l_days=600  #how many days to use in optimisations
res = pd.DataFrame(columns=['trades', 'momentum_window', 'minimum_momentum', 'portfolio_size',
                            'tr_period', 'cutoff', 'tot_contribution', 'final port_value',
                            'cumprod', 'tot_ret', 'drawdown'])
# total backtests 4*5*4*3*2=480
for bt in [100]:
    bt_days = l_close_min-bt
    for momentum_window in range(90, 540, 30):
        for minimum_momentum in range(70, 160, 10):
            for portfolio_size in [5, 10, 15, 20]:
                for cutoff in [0.01]:
                    for tr_period in [5, 10, 20]:
                        allocation = {}
                        dataset = len(df)-bt_days  # start for length of days used for the optimising dataset
                        l_days = momentum_window  # how many days to use in optimisations
                        port_value = 50000
                        non_trading_cash = 0
                        new_port_value = 0
                        print(bt, momentum_window, minimum_momentum,
                              portfolio_size, cutoff, tr_period)
                        added_value = tr_period*0
                        rs = backtest_portfolio(df, dataset, l_days, momentum_window, minimum_momentum, portfolio_size,
                                                tr_period, cutoff, port_value, added_value)
                        res = res.append(rs, ignore_index=True)
                        print(rs)

                        print(res.sort_values(by=['tot_ret']).tail(2))
print(res.sort_values(by=['tot_ret']).tail(20))
print(res.sort_values(by=['drawdown']).head(20))
best_res = res.sort_values(by=['tot_ret']).tail(1).reset_index(drop=True)
print(best_res)

'''
# show the backtest of the best portfolio
port_value = 50000
momentum_window = int(best_res.loc[0, 'momentum_window'])
minimum_momentum = int(best_res.loc[0, 'minimum_momentum'])
portfolio_size = int(best_res.loc[0, 'portfolio_size'])
cutoff = best_res.loc[0, 'cutoff']
tr_period =5# int(best_res.loc[0, 'tr_period'])
# how much cash to add each trading period
added_value = tr_period*0  # how much cash to add each trading period
no_tr = 1  # number of trades performed
allocation = {}
non_trading_cash = 0
init_portvalue = port_value
plotted_portval = []
plotted_ret = []
pval = pd.DataFrame(columns=['Date', 'portvalue', 'porteff'])
keep_df_buy = True
for days in range(dataset, len(df), tr_period):
    df_tr = df.iloc[days-l_days:days, :]
    df_date = datetime.strftime(df.iloc[days, :].name, '%d-%m-%Y')
    if days <= dataset:
        ini_date = df_date
    if days > dataset and not keep_df_buy:
        latest_prices = get_latest_prices(df_tr)
        new_port_value = non_trading_cash
        allocation = df_buy['shares'][:-1].to_dict()
        # print(allocation)
        if not keep_df_buy:
            #print('Sell date',df_date)
            for s in stocks:
                if s in allocation:
                    new_port_value = new_port_value + \
                        allocation.get(s)*latest_prices.get(s)
                    print('Sell ', s, 'stocks: ', allocation.get(s), ' bought for ', df_buy['price'][s], ' sold for ', latest_prices.get(s), ' for total:{0:.2f} and a gain of :{1:.2f}'.format(allocation.get(s)*latest_prices.get(s),
                          (latest_prices.get(s)-df_buy['price'][s])*allocation.get(s)))
            eff = new_port_value/port_value-1
            print('Return after trading period {0:.2f}%  for a total Value {1:.2f}'.format(
                eff*100, new_port_value))
            port_value = new_port_value
            plotted_portval.append(round(port_value, 2))
            plotted_ret.append(round(eff*100, 2))
            pval = pval.append({'Date': df_date,
                                'portvalue': round(port_value, 2),
                                'porteff': round(eff*100, 2)},
                               ignore_index=True)
            port_value = port_value+added_value  # add 200 after each trading period

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
    # print('universe',universe)
    # Create a df with just the stocks from the universe
    if len(universe) > 2:
        keep_df_buy = False
        df_t = select_columns(df_tr, universe)
        mu = capm_return(df_t)
        S = CovarianceShrinkage(df_t).ledoit_wolf()
        # Optimise the portfolio for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S)  # Use regularization (gamma=1)
        weights = ef.min_volatility()
        #weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights(cutoff)
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
            mom.append(df_m[df_m['stock'] == symbol].values[0])
            w.append(cleaned_weights[symbol])
            num_shares_list.append(num_shares)
            l_price.append(latest_prices[symbol])
            tot_cash.append(num_shares*latest_prices[symbol])

        df_buy = pd.DataFrame()
        df_buy['stock'] = symbol_list
        df_buy['momentum'] = mom
        df_buy['weights'] = w
        df_buy['shares'] = num_shares_list
        df_buy['price'] = l_price
        df_buy['value'] = tot_cash
        df_buy = df_buy.append({'stock': 'CASH',
                                'momentum': 0,
                                'weights': round(1-df_buy['value'].sum()/port_value, 2),
                                'shares': 0,
                                'price': 0,
                                'value': round(port_value-df_buy['value'].sum(), 2)},
                               ignore_index=True)
        df_buy = df_buy.set_index('stock')
        print('Buy date', df_date)
        print(df_buy)
        print('trade no:', no_tr, ' non allocated cash:{0:.2f}'.format(
            non_trading_cash), 'total invested:', df_buy['value'].sum())
        no_tr = no_tr+1
    else:
        print('Buy date', df_date,
              'Not enough stocks in universe to create portfolio', port_value)
        port_value = port_value+added_value
        keep_df_buy = True

total_ret = 100*(new_port_value/(init_portvalue+no_tr*added_value)-1)
dura = (no_tr-1)*tr_period
if no_tr > 2:
    print('Total return: {0:.2f} in {1} days'.format(total_ret, dura))
    print('Cumulative portfolio return:', round(
        list(pval['porteff'].cumsum())[-1], 2))
    print('total capital:', init_portvalue+no_tr*added_value, new_port_value)
    tot_contr = init_portvalue+no_tr*added_value
    s = round(pd.DataFrame(plotted_portval).pct_change().add(1).cumprod()*10, 2)
    rs = {'trades': no_tr,
          'momentum_window': momentum_window,
          'minimum_momentum': minimum_momentum,
          'portfolio_size': portfolio_size,
          'tr_period': tr_period,
          'cutoff': cutoff,
          'tot_contribution': tot_contr,
          'final port_value': new_port_value,
          'cumprod': s[-1:][0].values[0],
          'tot_ret': total_ret,
          'drawdown': s.diff().min()[0]}
    res = res.append(rs, ignore_index=True)
    print(rs)
    plt.plot(plotted_portval)

# create final and new portfolio
df_old = pd.read_csv('greekstocks_portfolio.csv').iloc[:, 1:]
cash = df_old.loc[df_old['stock'] == 'CASH']['value'].values[0]
new_price = []
for symbol in df_old['stock'][:-1].values.tolist():
    new_price.append(latest_prices[symbol])
new_price.append(0)
df_old['new_prices'] = new_price
df_old['new_value'] = df_old['new_prices']*df_old['shares']
port_value = 0.8*(cash+df_old['new_value'].sum())
# port_value=2200
df_tr = close_data.tail(l_days)
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
df_t = select_columns(df_tr, universe)
# portfolio
mu = capm_returns(df_t)
#mu = expected_returns.mean_historical_return(df_t)
# S=risk_models.sample_cov(df_t)
S = CovarianceShrinkage(df_t).ledoit_wolf()
# Optimise the portfolio for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)  # , gamma=1 Use regularization (gamma=1)
#weights = ef.efficient_return(sug_ret/100)
weights = ef.min_volatility()
#weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights(cutoff=cutoff)
# Allocate
latest_prices = get_latest_prices(df_t)
da = DiscreteAllocation(cleaned_weights,
                        latest_prices,
                        total_portfolio_value=port_value
                        )
allocation = da.greedy_portfolio()[0]
non_trading_cash = da.greedy_portfolio()[1]
ppf = list(ef.portfolio_performance())
print('Portfolio from the Assets that gave signals')
print('The proposed portfolio has the below characteristics')
print('Initial Portfolio Value : '+str(port_value)+'$')
print('Sharpe Ratio: '+str(round(ppf[2], 2)))
print('Portfolio Return: '+str(round(ppf[0]*100, 2))+'%')
print('Portfolio Volatility: '+str(round(ppf[1]*100, 2))+'%')


# Put the stocks and the number of shares from the portfolio into a df
symbol_list = []
mom = []
w = []
num_shares_list = []
l_price = []
tot_cash = []
for symbol, num_shares in allocation.items():
    symbol_list.append(symbol)
    mom.append(df_m[df_m['stock'] == symbol].values[0])
    w.append(cleaned_weights[symbol])
    num_shares_list.append(num_shares)
    l_price.append(latest_prices[symbol])
    tot_cash.append(num_shares*latest_prices[symbol])

df_buy = pd.DataFrame()
df_buy['stock'] = symbol_list
df_buy['momentum'] = mom
df_buy['weights'] = w
df_buy['shares'] = num_shares_list
df_buy['price'] = l_price
df_buy['value'] = tot_cash
df_buy = df_buy.append({'stock': 'CASH',
                        'momentum': 0,
                        'weights': round(1-df_buy['value'].sum()/port_value, 2),
                        'shares': 0,
                        'price': 0,
                        'value': round(port_value-df_buy['value'].sum(), 2)},
                       ignore_index=True)
df_buy = df_buy.set_index('stock')
print('Buy date', df_t.index[-1])
print(df_buy)
df_buy = df_buy.reset_index()
print(df_buy['value'].sum())

# rebalance with old portfolio
rebalance_portfolio(df_old, df_buy)
'''
