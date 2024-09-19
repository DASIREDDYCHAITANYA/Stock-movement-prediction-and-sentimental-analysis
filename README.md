# Stock-movement-prediction-and-sentimental-analysis
Overview:
This project aims to predict Tesla stock price movements by analyzing sentiment data scraped from social media platforms like Twitter, Reddit, and Telegram. By merging this sentiment data with historical Tesla stock prices, the model forecasts future price trends based on the correlation between user discussions and stock performance.

DATASET PREPARATION:
NOTE: I have prepared in twi ands merged with tesla historical dataset wiht equal dates 

-> SEntiment  Dataset creation using NLP
1. Downloading and Loading the Dataset
 code
!kaggle datasets download -d equinxx/stock-tweets-for-sentiment-analysis-and-prediction
import pandas as pd
data1 = pd.read_csv('stock_tweets.csv')
Downloads the dataset using Kaggle CLI.
Loads the CSV file into a DataFrame named data1.
2. Preparing Data for Sentiment Analysis
code:
sent_df = data1.copy()
sent_df["sentiment_score"] = ''
sent_df["Negative"] = ''
sent_df["Neutral"] = ''
sent_df["Positive"] = ''
sent_df.head()

Adds columns for sentiment scores: sentiment_score, Negative, Neutral, and Positive.

 Performing Sentiment Analysis
 
 code:

%%time
sentiment_analyzer = SentimentIntensityAnalyzer()
for indx, row in sent_df.iterrows():
    try:
        sentence = unicodedata.normalize('NFKD', row['Tweet'])
        sentiment = sentiment_analyzer.polarity_scores(sentence)
        sent_df.at[indx, 'sentiment_score'] = sentiment['compound']
        sent_df.at[indx, 'Negative'] = sentiment['neg']
        sent_df.at[indx, 'Neutral'] = sentiment['neu']
        sent_df.at[indx, 'Positive'] = sentiment['pos']
    except TypeError:
        print(row['Tweet'])
        print(indx)
        break
sent_df.head()

Initializes the sentiment analyzer.
Iterates over each row to analyze the tweet's sentiment. 
Assign and sentimental score according the tweet like negative ,nuteral and positve
Stores the sentiment scores in the DataFrame.
Measures execution time with %%time.

Calculating Daily Average Sentiment
code
data1['Date'] = pd.to_datetime(data1['Date'])
daily_average_sentiment = data1.groupby('Date').agg({'sentiment_score': 'mean'}).reset_index()
data = daily_average_sentiment
data

Converts Date column to datetime format.
Groups data by Date and computes the mean sentiment score.
Displays the DataFrame with daily average sentiment.data.
data: Displays the DataFrame with daily average sentiment scores.

-> sentiment Dataset using API
Here’s an overview of the script that fetches and processes sentiment data for Tesla (TSLA) using the Alpha Vantage API:

code:

HTML("""
<style>
.container { width:100% !important; }
</style>
""")
Imports: Includes necessary libraries such as requests for API calls, pandas for data manipulation, and time for managing API call limits.
HTML: Adjusts the width of the notebook container.

 API Key and Functions
code
api_key = 'H89X3Q1Y9FUJEJ99'

def _get_data(symbols, time_from, time_to, api_key):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbols}&time_from={time_from}&time_to={time_to}&limit=1000&apikey={api_key}"
    r = requests.get(url)
    data = r.json()
    return data

def _get_label_sentiment(x):
    if x <= -0.35:
        return 'Bearish', 'Bearish'
    elif -0.35 < x <= -0.15:
        return 'Somewhat-Bearish', 'Bearish'
    elif -0.15 < x < 0.15:
        return 'Neutral', 'Neutral'
    elif 0.15 <= x < 0.35:
        return 'Somewhat_Bullish', 'Bullish'
    else:  # x >= 0.35
        return 'Bullish', 'Bullish'

API Key: Stores the Alpha Vantage API key.
_get_data Function: Fetches news sentiment data from the Alpha Vantage API for given symbols, time range, and API key.
_get_label_sentiment Function: Classifies sentiment scores into categories like 'Bearish', 'Neutral', or 'Bullish'.

 Data Collection and Processing
code:
def get_dataset(time_from="20230410T0130", time_to='', MAX_API_CALLS_PER_DAY = 25, MAX_API_CALLS_PER_MIN = 5):
    data_list = []
    for i in range(1, MAX_API_CALLS_PER_DAY + 1):
        if i % 5 == 0:
            time.sleep(60)

        data = _get_data('TSLA', time_from, time_to, api_key)
        if 'feed' not in data: break
        if len(data['feed']) == 0: break
        data_list.append(data)
        time_to = data['feed'][-1]['time_published'][:-2]  # Update time_to for the next call

    df = pd.concat([pd.DataFrame(data['feed']) for data in data_list])
    
    # Extract TSLA specific data
    df['ticker_relevance_TSLA'] = df['ticker_sentiment'].apply(lambda l: [el for el in l if el['ticker'] == 'TSLA'][0]['relevance_score']).astype(float)
    df['ticker_sentiment_TSLA'] = df['ticker_sentiment'].apply(lambda l: [el for el in l if el['ticker'] == 'TSLA'][0]['ticker_sentiment_score']).astype(float)
    
    # Filter for TSLA in headlines
    df = df[df.title.str.contains('tsla|tesla', case=False)]
    
    # Count number of tickers and filter for single ticker
    df['num_tickers'] = df.ticker_sentiment.apply(lambda l: len(l))
    df = df[df.num_tickers == 1]
    
    # Apply sentiment labeling
    df[['detailed_original_label', 'label']] = df.apply(lambda row: _get_label_sentiment(row['ticker_sentiment_TSLA']), axis=1, result_type='expand')
    
    # Drop duplicates and sort
    df.drop_duplicates(subset=['summary'], inplace=True, keep='first')
    df.set_index('time_published', inplace=True)
    df.sort_index(inplace=True)
    
    return df
get_dataset Function:

Calls _get_data in a loop, managing API call limits and delays.
Aggregates data into a single DataFrame.
Extracts TSLA-specific sentiment and relevance scores.
Filters rows containing TSLA in headlines.
Further filters to ensure only single ticker mentions.
Applies sentiment labels and cleans the DataFrame by removing duplicates and sorting.

 Saving and Displaying Data
 code
df = get_dataset(time_to='')
df.to_csv('tsla_sentiment.csv')
df

Fetch Data: Calls get_dataset to retrieve and process data.
Save to CSV: Saves the DataFrame to tsla_sentiment.csv.
Display DataFrame: Displays the DataFrame with the sentiment data.

This script effectively collects and processes sentiment data related to Tesla, ensuring API rate limits are adhered to and resulting data is filtered and cleaned for further analysis.
NOTE: Now i have merged two datsets to tarin model efficiently

-> Historical dataset from yahoo:
code:
import yfinance as yf
data2= yf.download("TSLA  ", start="2021-09-30", end="2024-09-16")

This code retrives historical  dataset from yfinance platfrom on TSLA stock.

-> Final dataset:
code:
import pandas as pd
dataset1 = pd.read_csv(dataset1)
dataset2 = pd.read_csv(dataset2)
dataset1['time_published'] = pd.to_datetime(dataset1['time_published'])

dataset1 = dataset1.groupby('time_published').agg({'overall_sentiment_score': 'mean'}).reset_index()

I have loaded my all datasets which i have merged and historical dataset

code:

merged_df['time_published'] = pd.to_datetime(merged_df['time_published'])
df3['Date'] = pd.to_datetime(df3['Date'])
final_df = pd.merge(merged_df, df3, left_on='time_published',right_on="Date")

Now i have merged all features to a single dataset  with equal dates

NOTE: I have processed Data of past 4 years . At last i got only 500 records because some authentication problems

-> MODEL TRAINIGL:
