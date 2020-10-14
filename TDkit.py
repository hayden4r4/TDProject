from td.client import TDClient
import seaborn as sns
import pandas as pd
import numpy as np
import os
import edhec_risk_kit as erk
import operator
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import quandl as ql
import pandas_market_calendars as mcal

consumer_key=open('API KEYS\TD_CONSUMER_KEY.txt', 'r').read()
redirect_uri='http://localhost'
credentials_path='Credential.json'
api_key=open('API KEYS\TD_ACCESS_TOKEN.txt', 'r').read()
td_client=TDClient(client_id=consumer_key, redirect_uri=redirect_uri, credentials_path=credentials_path)    

Ticker_List={'Tickers':['SPY', 
                        'AAPL',
                        'AMZN', 
                        'NVDA',
                        'AMD',
                        'MSFT',
                        'GOOGL',
                        'SOXX',
                        'JD',
                        'SNE',
                        'TSLA',
                        'CHGG']}
n_assets=len(Ticker_List['Tickers'])

weights=                [.1672,
                         .2185,
                         .1680,
                         .0867,
                         .0650,
                         .0768,
                         .0716,
                         .0531,
                         .0293,
                         .0287,
                         .0178,
                         .0172]
weights_dic=dict(zip(Ticker_List['Tickers'], weights))



def login():
    """
    Logs in to TD Ameritrade.  MUST BE RAN FIRST
    """
    td_client.login()


def Closing_price_alltime(Ticker, n_years):
    """
    Gets Closing Price for Past 20 Years w/ Daily Intervals
    and Formats it to correct Date and single 'Closing Price'
    column.
    """
    Closedf=td_client.get_price_history(Ticker, period_type='year', period=n_years, frequency_type='daily', frequency=1)
    return Closedf


def pct_change(Ticker_ClosingValues):
    """
    Takes Closing Values and Finds Percent Change.
    Closing Value Column must be named 'Closing Price'.
    """
    return_pct=Ticker_ClosingValues.pct_change()
    return_pct=return_pct.dropna()
    return return_pct


def Port_consol(Ticker_List, n_years):
    """
    Consolidates Ticker Symbol Returns and Returns
    a Single Portolio
    """
    Port=[]
    Port_=[]
    for i in Ticker_List['Tickers']:
        Port.append(Closing_price_alltime(i, n_years))
    j=list(range(0, (n_assets)))
    for i in j:
        data = operator.itemgetter('datetime','close')
        symbol = Port[i]['symbol']
        candles = Port[i]['candles']
        dt, closing = zip(*map(data, candles))
        s = pd.Series(data=closing,index=dt,name=symbol)
        s=pd.DataFrame(s)
        s.index = pd.to_datetime(s.index, unit='ms')
        Port_.append(s)
    Portfolio=pd.concat(Port_, axis=1, sort=False)
    Portfolio.index.names=['Date']
    return Portfolio

    
def ql_rf(n_days):
    """
    Pulls 10-Year Treasury Bond Rate from US Treasury
    using Quandl API for use as Risk-Free Rate.
    """
    days_ago = datetime.now() - relativedelta(days=n_days)
    days_ago=days_ago.date()
    qlapi=open('API KEYS/QUANDL_API_KEY.txt', 'r').read()
    riskfree_rate=ql.get("USTREASURY/YIELD.10", start_date=days_ago, authtoken=qlapi)
    riskfree_rate=riskfree_rate.dropna()
    return riskfree_rate


def Portfolio_rets(Weighted_port):
    """
    Cleans Up Portfolio Returns Which are Calculated from the
    Weighted Portfolio Returns.
    """
    Port_rets=Weighted_port.sum(axis=1)
    Port_rets=pd.DataFrame(Port_rets)
    Port_rets=Port_rets.rename(columns={0:'Returns'})
    return Port_rets


def timeslice(Port_rets, Start_Date, End_Date=None):
    """
    Slices Time for Portfolio Returns.
    """
    Port_rets=Port_rets.loc[pd.IndexSlice[Start_Date:End_Date]]
    return Port_rets


def Get_and_cleanup(Ticker, Quote_type='close', n_years=20):
    """
    Pulls Specified Stock's Data for Specified Time Frame
    & Keeps Only Specified Quote Type, then Cleans Up and
    Returns DataFrame.
    """
    df=td_client.get_price_history(Ticker, period_type='year', period=n_years, frequency_type='daily', frequency=1)
    data = lambda df:(df['datetime'], df[Quote_type])
    symbol = df['symbol']
    candles = df['candles']
    dt, closing = zip(*map(data, candles))
    s = pd.Series(data=closing,index=dt,name=symbol)
    s=pd.DataFrame(s)
    s.index = pd.to_datetime(s.index, unit='ms')
    s.index.names=['Date']
    return s


def get_market_calendar(yrs_ago):
    """
    Gets Days That NYSE is Open Between Yesterday and Specified
    Number of Years Ago.
    """
    nyse = mcal.get_calendar('NYSE')
    start_date=datetime.now() - timedelta(days=yrs_ago*365)
    start_date=datetime.strftime(start_date, '%Y-%m-%d')
    end_date=datetime.now() - timedelta(days=1)
    end_date=datetime.strftime(end_date, '%Y-%m-%d')
    mkt_cal=nyse.valid_days(start_date=start_date, end_date=end_date)
    return len(mkt_cal)


def annualize_rets(r, periods_per_year):
    """
    Annualizes a Set of Returns.
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def spy_ann(n_years=20):
    """
    Annualizes the Return of SPY.
    """
    SPY=Get_and_cleanup('SPY', n_years=n_years)
    n_mkt_days=get_market_calendar(n_years)
    n_mkt_days_yr=n_mkt_days/n_years
    SPY_pct=SPY.pct_change()
    SPY_ann=annualize_rets(r=SPY_pct, periods_per_year=n_mkt_days_yr)
    return SPY_ann


def Port_holdings_ann(Port_pct, n_years=20):
    Holdings_ann=[]
    for i in Ticker_List['Tickers']:
        n_mkt_days=get_market_calendar(n_years)
        n_mkt_days_yr=n_mkt_days/n_years
        r=Port_pct[i]
        s=annualize_rets(r, periods_per_year=n_mkt_days_yr)
        Holdings_ann.append(s)
        Port_holdings_ann_rets=dict(zip(Ticker_List['Tickers'], Holdings_ann))
        Port_holdings_ann_rets=pd.DataFrame(Port_holdings_ann_rets, index=[0])
    return Port_holdings_ann_rets


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std()*(periods_per_year**0.5)


def Port_holdings_ann_vol(Port_pct, n_years=20):
    Holdings_ann_vol=[]
    for i in Ticker_List['Tickers']:
        n_mkt_days=get_market_calendar(n_years)
        n_mkt_days_yr=n_mkt_days/n_years
        r=Port_pct[i]
        s=annualize_vol(r, periods_per_year=n_mkt_days_yr)
        Holdings_ann_vol.append(s)
        Port_holdings_ann_vol_rets=dict(zip(Ticker_List['Tickers'], Holdings_ann_vol))
        Port_holdings_ann_vol_rets=pd.DataFrame(Port_holdings_ann_vol_rets, index=[0])
    return Port_holdings_ann_vol_rets


def optimal_weights(n_points, er, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5



from scipy.optimize import minimize

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x



def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
        return ax


        
    
    
    
