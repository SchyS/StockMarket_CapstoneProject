# StockMarket_CapstoneProject
## Project Overview
The goal of this project is to classify whether stock trades will be profitable or unprofitable based on features beyond 
the specified trading strategy. Although the stock market is sensitive to many variables both numeric and text, this 
project focuses on the numeric side. This project will incorporate rule-based trading strategies with a pipeline across
several models to determine classify the outcome.

## Dataset Description
The data acquisition process used Yahoo Finance's API, yfinance, to collect the S&P 500 ETF (Ticker: SPY) Data over
the last 15 years (1/03/2011 to 12/31/2025). 

The raw S&P500 ETC dataset goes through several preprocessing steps. First, missing values are checked in each by looking
for null and missing data. None are found. Next, duplicates were checked by first looking for duplicate dates, then by looking 
for exact 1 to 1 identical rows. No duplicates were found. 

Our next step was feature engineering. Several core features are used and have variances depending on the length (e.g. simple 
moving average over 10 days = SMA_10). Here are the list of our features:

entry_date, exit_date, strategy, pnl_price, profitable, Date, Close, High, Low, Open, Volume, prev_close, prev_volume, ret_1d, log_ret_1d, 
gap_retm, mom_5d, mom_10d, mom_20d, SMA_20, SMA_50, SMA_200, vol_20d, vol_20d_ann, Vol_SMA_20, rel_volume_20, OBV, VWAP, CMF_20, 
close_vs_sma20, close_vs_sma50, close_vs_sma200, sma20_vs_sma50, sma50_vs_sma200, sma20_vs_sma200, zclose_20, EMA_12, EMA_26, MACD, 
MACD_signal, MACD_hist, RSI_14, ATR_14, BB_mid_20, BB_std_20, BB_upper_20, BB_lower_20, BB_width_20, BB_pctB_20, stoch_k_14, stoch_d_3

## Methology
After feature engineering we must incorporate and start building our pipeline starting with the trading strategies. The following four 
strategies are used:

Close and Simple Moving Average over 200 Days: Enter when RSI_14 is less than 40 AND when the "Close" is less than the Simple Moving 
Average over 200 Days. Exit after 7 days or when a 0.02 is gained or lost.

Relative Strength Index: Enter when the Relative Strength Index is less than 40. Exit after 7 days or when a 0.02 is gained or lost.

Simple Moving Average 20 Days & Simple Moving Average 50 Days: Enter when the Simple Moving Average 20 Days crosses below the 
Simple Moving Average 50 Days. Exit after 7 days, when a 0.02 is gained or lost, or when SMA_50 crosses back above SMA_20.

Simple Moving Average 20 Days, 50 Day, 200 Days: Enter when Simple Moving Average 200 Days is less than Simple Moving Average 50 Days
and Simple Moving Average 50 Days is less than Simple Moving Average 20 Days.  Exit after 7 days, when a 0.02 is gained or lost, or when 
Simple Moving Average 50 Days crosses above Simple Moving Average 20 Days.

This pipeline executes the trades, automatically stores the trade information within a table and classifies them into whether they were 
profitable or non profitable. 

With this new dataset containing the engineered features, trading information and orginal dataset features we train our machine learning
models.

*Note the the features: entry_date, exit_date, Date, Close, High, Low, Open, Volume, and profitable are excluded due to giving future 
information that would not be provided at the time the trade is made in a real-world scenario. Profitable is excluded because it is
the target feature.

Three models are used: Logistical Regression, Random Forest and a Neural Network. Each target variable is "profitable" and each model
is trained on each type of strategy on its own. For example, Relative Strength Index classifier is only trained on the Relative Strength
Index strategy data (trades).

Lastly, the results are both stored within Databricks MLFlow Experiments and Outputs as csv files. 

## Run Instructions
*Please note to change file paths to your correct paths
First download the stock data using yFinace.

Open "CapstoneProject Extened" Folder.

Go to "Data Preparation" and run each cell.

Optional: Go and Run Milestone 3 for Exploratory Data Analysis.

Go to "Trading Strategies" and run each cell.

Go into "Milestone 4" Folder. 

Go into "Machine Learning Models" Folder.

For each of the three model files, "Logisitic Regression", "Neural Network", and "Random Forest", and run each cell in each file.

Following these steps should give the outputs for each model and are automatically saved as csv files. There are several other
files included but are optional in running. These include but are not limited to exploratory data analysis, model comparisons
and the original result files.
