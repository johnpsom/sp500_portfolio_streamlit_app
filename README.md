# SP500_portfolios_Streamlit_App
A Streamlit App to create and test portfolios of SP500 stocks

This Streamlit App is an attempt to create and research my own trading strategy based on Portfolio creation and testing from the SP500 stocks universe.
Basically it uses the PyPortfolioOpt module to create portfolios but from stocks that in the recent past days have shown a certain momentum in increasing their price.

After fetching enough historical data using the yahoo_fin (to get our stocks symbols) and yfinance modules, we choose our portfolio initial
capital or the $$$ we would like to invest. 
Then we decide on the momentum window which is the number of days we use to calculate our momentun indicator.
And finally we choose a minimum value of momentum that a stock must have to be included in the stock universe from which we create our portfolio.
It uses also a max portfolio size in the case that we have plenty of stocks that satisfy the above resrtictions.
Then we choose the minimum percentage of each stock in our portfolio.

The trading period slider decides on the rebalancing period of days and is used for the backtesting of our portfolio. Basically what it does it uses the
chosen parameters to create a portfolio and the days of the backtest. So it goes back in the historical values and uses them to see what would have happened
if we used a portfolio of the same paramaters for these last number of days. 
It does that by rebalancing our portfolio for three rebalancing periods which are every week or 5 days, every fortnight or 10 days and every month or 20 days,
and getd the redults.

For me this works as that if we find that have a satisfying backtest we hope that for the same period as the rebalancing period used we are going to have a 
same result and our portfolio will increase in value.
There are no trading costs included but what you can also do, is to fund your initial capital with a certain amount say of 10$ per day and see what happens then.
For example 50$ per days means that in a trading period of 20 days you add 1000$ each month or for a trading period of 5 days you add to you capital 250$ per week.

In my git hub repository there is another file which you can get freely which does a brute search for all the combinations of the parameters used for the backtest.

**As stated on the App's Page this is not a trading proposal and what is presented here is for educational and informational purposes only and in no case 
the creator of this App can be held responsible for any loss of money by using this App.** Each one of us is the only person responsible for our trading decisions.


