# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:00:53 2021

@author: johnpsom
"""

import yfinance as yf
import yahoo_fin.stock_info as si


def stocks_list():
    """ get all the current symbols of the SP500 stocks """
    return si.tickers_sp500()


# pull data for each S&P stock
def download_from_yahoo(tickers, start, end, interval):
    '''tickers is a list of the tickers we want to download stock prices for
       start and end are datetime objects specifying the time range to download
       interval is typically '1d', indicating we want daily data '''
    print('downloading {} stocks from {} to {}'.format(len(tickers), start, end))
    _data = yf.download(tickers, start=start, end=end, interval=interval)
    return _data
