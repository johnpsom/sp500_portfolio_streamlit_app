# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:00:47 2021

@author: johnpsom
"""

import base64
import io
import json
import pickle
import re
import uuid
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import capm_return
from pypfopt.risk_models import CovarianceShrinkage
from scipy import stats


# Momentum score function
def momentum_score(ts):
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    regress = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(regress[0]), 252) - 1) * 100
    return annualized_slope * (regress[2] ** 2)


# select stocks columns
def select_columns(data_frame, column_names):
    new_frame = data_frame.loc[:, column_names]
    return new_frame


# cumulative returns calculation
def cumulative_returns(stock, returns):
    res = (returns + 1.0).cumprod()
    res.columns = [stock]
    return res


def get_portfolio(universe, df_tr, port_value, cutoff, df_m):
    """create a portfolio using the stocks from the universe and the closing
    prices from df_tr with a given portfolio value and a weight cutoff value
    using the value of a momentum indicator to limit the quantity of the stocks"""
    df_t = select_columns(df_tr, universe)
    mu = capm_return(df_t)
    S = CovarianceShrinkage(df_t).ledoit_wolf()
    # Optimize the portfolio for minimum volatility or maximum Sharpe ratio
    ef = EfficientFrontier(mu, S)  # Use regularization (gamma=1)
    weights = ef.min_volatility()
    # weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights(cutoff=cutoff)
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
        tot_cash.append(num_shares * latest_prices[symbol])

    df_buy = pd.DataFrame()
    df_buy['stock'] = symbol_list
    df_buy['momentum'] = mom
    df_buy['weights'] = w
    df_buy['shares'] = num_shares_list
    df_buy['price'] = l_price
    df_buy['value'] = tot_cash
    df_buy = df_buy.append(
        {'stock': 'CASH', 'momentum': 0, 'weights': round(1 - df_buy['value'].sum() / port_value, 2), 'shares': 1,
         'price': round(port_value - df_buy['value'].sum(), 2), 'value': round(port_value - df_buy['value'].sum(), 2)},
        ignore_index=True)
    df_buy = df_buy.set_index('stock')
    return df_buy


def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)
            towrite = io.BytesIO()
            # object_to_download = object_to_download.to_excel(towrite, encoding='utf-8', index=False, header=True)
            towrite.seek(0)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(towrite.read()).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)
    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + \
        f'<a download="{download_filename}" id="{button_id}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">{button_text}</a><br></br>'
    return dl_link


def backtest_portfolio(df, dataset=1000, l_days=700, momentum_window=120, minimum_momentum=70, portfolio_size=5,
                       tr_period=5, cutoff=0.05, port_value=10000, a_v=0):
    allocation = {}
    non_trading_cash = 0
    new_port_value = 0
    stocks = df.columns
    added_value = tr_period * a_v
    no_tr = 1  # number of trades performed
    init_portvalue = port_value
    plotted_portval = []
    plotted_ret = []
    pval = pd.DataFrame(columns=['Date', 'portvalue', 'porteff'])
    keep_df_buy = True
    for days in range(dataset, len(df), tr_period):
        df_tr = df.iloc[days - l_days:days, :]
        df_date = datetime.strftime(df.iloc[days, :].name, '%d-%m-%Y')
        if days <= dataset:
            ini_date = df_date
            plotted_portval.append(round(port_value, 2))
            plotted_ret.append(0)
        if days > dataset and keep_df_buy is False:
            latest_prices = get_latest_prices(df_tr)
            new_port_value = non_trading_cash
            allocation = df_buy['shares'][:-1].to_dict()
            # print(allocation)
            if keep_df_buy is False:
                # print('Sell date',df_date)
                for s in allocation:
                    new_port_value = new_port_value + \
                        allocation.get(s) * latest_prices.get(s)
                    # print('Sell ',s,'stocks: ',allocation.get(s),' bought for ',df_buy['price'][s],' sold for ',latest_prices.get(s)
                    #       ,' for total:{0:.2f} and a gain of :{1:.2f}'.format(allocation.get(s)*latest_prices.get(s),
                    #      (latest_prices.get(s)-df_buy['price'][s])*allocation.get(s)))
                eff = new_port_value / port_value - 1
                # print('Return after trading period {0:.2f}%  for a total Value {1:.2f}'.format(eff*100,new_port_value))
                port_value = new_port_value
                plotted_portval.append(round(port_value, 2))
                plotted_ret.append(round(eff * 100, 2))
                pval = pval.append({'Date': df_date, 'portvalue': round(port_value, 2), 'porteff': round(eff * 100, 2)},
                                   ignore_index=True)
                port_value = port_value + added_value  # add 200 after each trading period
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
        df_m = df_m[
            (df_m['momentum'] > minimum_momentum - 0.5 * dev) & (df_m['momentum'] < minimum_momentum + 1.9 * dev)].head(
            portfolio_size)
        # Set the universe to the top momentum stocks for the period
        universe = df_m['stock'].tolist()
        # print('universe',universe)
        # Create a df with just the stocks from the universe
        if len(universe) > 2 and port_value > 0:
            keep_df_buy = False
            df_buy = get_portfolio(universe, df_tr, port_value, cutoff, df_m)
            # print('Buy date',df_date)
            # print(df_buy)
            # print('trade no:',no_tr,' non allocated cash:{0:.2f}'.format(non_trading_cash),'total invested:', df_buy['value'].sum())
            port_value = df_buy['value'].sum()
            no_tr = no_tr + 1
            # st_day=st_day+tr_period
        else:
            # print('Buy date',df_date,'Not enough stocks in universe to create portfolio',port_value)
            port_value = port_value + added_value
            keep_df_buy = True

    total_ret = 100 * (new_port_value /
                       (init_portvalue + no_tr * added_value) - 1)
    dura_tion = (no_tr - 1) * tr_period
    if no_tr > 2:
        # print('Total return: {0:.2f} in {1} days'.format(total_ret,dura_tion))
        # print('Cumulative portfolio return:',round(list(pval['porteff'].cumsum())[-1],2))
        # print('total capital:',init_portvalue+no_tr*added_value, new_port_value)
        tot_contr = init_portvalue + no_tr * added_value
        s = round(pd.DataFrame(plotted_portval).pct_change().add(
            1).cumprod() * 10, 2)
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

        return rs, pval


def rebalance_portfolio(df_old, df_new):
    '''rebalance old with new proposed portfolio'''
    old_port_value = df_old['value'].sum()
    new_port_value = old_port_value
    new_stocks = list(df_old.stock[:-1]) + \
        list(set(df_new.stock[:-1]) - set(df_old.stock))
    for stock in new_stocks:
        # close old positions that do not appear in new portfolio
        if (stock in list(df_old.stock)) and (stock not in list(df_new.stock[:-1])):
            # close positions of stocks that do not exist in the new portfolio
            if df_old.loc[df_old.stock == stock, 'shares'].values[0] > 0:
                st.write(f'Close your position on {stock}')
                new_port_value = new_port_value + \
                    df_old.loc[df_old.stock == stock, 'shares'].values[0]
            if df_old.loc[df_old.stock == stock, 'shares'].values[0] < 0:
                st.write(f'Close your position on {stock}')
                new_port_value = new_port_value + \
                    df_old.loc[df_old.stock == stock, 'shares'].values[0]
        # open new positions that only appear in new portfolio
        if stock in list(set(df_new.stock[:-1]) - set(df_old.loc[:, 'stock'])):
            if df_new.loc[df_new.stock == stock, 'shares'].values[0] > 0:
                st.write(
                    f'Buy  {df_new.loc[df_new.stock == stock, "shares"].values[0]} shares of {stock} to open new long position')
            if df_new.loc[df_new.stock == stock, 'shares'].values[0] < 0:
                st.write(
                    f'Sell short {df_new.loc[df_new.stock == stock, "shares"].values[0]} shares of {stock} to open new short posotion')
        # modify positions of stocks that appear in new and old portfolio
        if (stock in list(df_old.stock)) and (stock in list(df_new.stock[:-1])):
            # change positions
            if df_new.loc[df_new.stock == stock, 'shares'].values[0] > 0 and \
                    df_old.loc[df_old.stock == stock, 'shares'].values[0] > 0:
                new_shares = df_new.loc[df_new.stock == stock, "shares"].values[0] - \
                    df_old.loc[df_old.stock == stock, 'shares'].values[0]
                if new_shares >= 0:
                    st.write(
                        f'Buy another {round(new_shares, 0)} shares of stock {stock}')
                if new_shares < 0:
                    st.write(
                        f'Sell another {round(-new_shares, 0)} shares of stock {stock}')
            if df_new.loc[df_new.stock == stock, 'shares'].values[0] < 0 and \
                    df_old.loc[df_old.stock == stock, 'shares'].values[0] < 0:
                new_shares = df_new.loc[df_new.stock == stock, "shares"].values[0] - \
                    df_old.loc[df_old.stock == stock, 'shares'].values[0]
                if new_shares >= 0:
                    st.write(
                        f'Buy another {round(new_shares, 0)} shares of stock {stock}')
                if new_shares < 0:
                    st.write(
                        f'Sell another {round(-new_shares, 0)} shares of stock {stock}')
            if df_new.loc[df_new.stock == stock, 'shares'].values[0] * \
                    df_old.loc[df_old.stock == stock, 'shares'].values[0] < 0:
                new_shares = df_new.loc[df_new.stock == stock, 'shares'].values[0] - \
                    df_old.loc[df_old.stock == stock, 'shares'].values[0]
                if new_shares >= 0:
                    st.write(
                        f'Buy another {round(new_shares, 0)} shares of stock {stock}')
                if new_shares < 0:
                    st.write(
                        f'Sell another {round(-new_shares, 0)} shares of stock {stock}')
    return new_port_value
