# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:10:58 2019

@author: Henry
"""

import pandas as pd

df1 = pd.read_csv('articles1.csv')
df2 = pd.read_csv('articles2.csv')
df3 = pd.read_csv('articles3.csv')

df1.drop(columns = ['Unnamed: 0','id'], inplace= True)
df2.drop(columns = ['Unnamed: 0','id'], inplace= True)
df3.drop(columns = ['Unnamed: 0','id'], inplace= True)

df = df1.append([df2,df3])

df.content = df.content.astype('str')
#142570

df_clean = df.drop_duplicates('content','first')
#142038

df_clean.date = pd.to_datetime(df_clean.date)
df_clean.groupby(df_clean["date"].dt.year).publication.count().plot(kind="bar")
# Year with > 100 articles: 2013 thru 2017

# Subset data more to only include months with sufficient articles
data = df_clean[(df_clean.date <= '2017-06-30') & (df_clean.date >= '2016-01-01')]

data.set_index('date').groupby(pd.Grouper(freq='M')).publication.count().plot(kind="bar")


df = data.reset_index(drop=True)
df = df[['title','publication','author','date','content']]

print(len(df))
#135097 rows
#---------------------------------------------------------------------------

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

sentiments = []
x = 0
for article in df.content:
    sentiments += [sid.polarity_scores(article)['compound']]
    if x % 100 == 0:
        print(x)
    x += 1
    
df['sentiments'] = sentiments

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import dates

def plot_sentiment(publication, df):
    pub = df[df.publication == publication]
    per_month = pub.set_index('date').groupby(pd.Grouper(freq='M'))['sentiments'] \
       .agg({'text':'size', 'sent':'mean'}) \
       .rename(columns={'text':'count','sent':'mean'}) 
    for index, row in per_month.iterrows():
        if row['count'] < 20:
            per_month['mean'][index] = None
    per_month = per_month[['mean']].reset_index()
    per_month.columns = ['Date',publication]
    return(per_month)
    
pubs = ['New York Times', 'Breitbart', 'CNN', 'Business Insider',
       'Atlantic', 'Fox News', 'Buzzfeed News',
       'National Review', 'New York Post', 'Guardian', 'NPR', 'Reuters',
       'Vox', 'Washington Post']
#Removed TalkingPointsMemo for too few articles
    
sentiment_by_pub = plot_sentiment('Washington Post', df)[['Date']]
for pub in pubs:
    sentiment_by_pub = sentiment_by_pub.merge(plot_sentiment(pub, df), how='outer')

plot_sent = sentiment_by_pub.melt('Date', var_name='Publication',  value_name='Sentiment')

sns.set(style="darkgrid", font_scale = 1)
fig,ax = plt.subplots(figsize = (8,5))
g = sns.lineplot(ax=ax, x="Date", y="Sentiment", hue='Publication', data=plot_sent, linewidth=2.5)
leg = plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 8})
for legobj in leg.legendHandles[1:]:
    legobj.set_linewidth(2.5)
#g.set(xticks = plot_sent['Date'].values)
ax.xaxis.set_major_formatter(dates.DateFormatter("%m-%Y"))
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig('Sentiment_by_Publication.pdf')

#-----------------------------------------------------------------
def overall_sentiment(df):
    per_month = df.set_index('date').groupby(pd.Grouper(freq='M'))['sentiments'] \
       .agg({'text':'size', 'sent':'mean'}) \
       .rename(columns={'text':'count','sent':'mean'}) 
    per_month = per_month[['mean']].reset_index()
    per_month.columns = ['Date', 'Average_Sentiment']
    return(per_month)
    
avg = overall_sentiment(df)
    
diffs = sentiment_by_pub
for col in diffs.columns[1:]:
    diffs[col] = diffs[col] - avg.Average_Sentiment

plot_diffs = diffs.melt('Date', var_name='Publication',  value_name='Sentiment')

sns.set(style="darkgrid", font_scale = 1)
fig,ax = plt.subplots(figsize = (8,5))
g = sns.lineplot(ax=ax, x="Date", y="Sentiment", hue='Publication', data=plot_diffs, linewidth=2.5)
leg = plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 8})
for legobj in leg.legendHandles[1:]:
    legobj.set_linewidth(2.5)
#g.set(xticks = diffs['Date'].values)
ax.xaxis.set_major_formatter(dates.DateFormatter("%m-%Y"))
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig('Standardized_Sentiment_by_Publication.pdf')


#diffs.to_csv('Average_Difference_Sentiment.csv')
#---------------------------------------------------------------------
sns.set(style="darkgrid", font_scale = 1)
fig,ax = plt.subplots(figsize = (8,5))
sns.lineplot(x="Date",y="Average_Sentiment",data=avg, linewidth=4)
ax.xaxis.set_major_formatter(dates.DateFormatter("%m-%Y"))
plt.xticks(rotation=45)
plt.ylim(0,.3)
plt.tight_layout()
fig.savefig('Average_Sentiment.pdf')


#--------------------------------------------------------

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
import nltk
from tqdm import tqdm

tqdm.pandas()

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_toks(text):
    clean = word_tokenize(text)
    clean = [word for word in clean if len(word)>2]
    clean = [word for word in clean if word not in en_stop]
    clean = [nltk.stem.porter.PorterStemmer().stem(word) for word in clean]
    clean = [word for word in clean if word not in ['the','said', 'say', 'like','every','many','thing',
                                                    'want','really','yet','just','one','would','new','also',
                                                    'say','thi','come','told','take','way','use','even','get',
                                                    'could','also','tri','ask','made','see','much','they','that',
                                                    'go','thi','told','way','sinc','still','may','000','still']]
    clean = [word for word in clean if word not in en_stop]
    return clean    

df['cleaned_text'] = df.content.progress_apply(lambda x: clean_toks(x))

df.to_csv('Cleaned_Data.csv')

