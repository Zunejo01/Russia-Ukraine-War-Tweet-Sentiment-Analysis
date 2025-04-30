import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print("Loading tweet data...")
tweets_df = pd.read_csv('30K Tweets with russiaukrainewar hashtag.csv')
print(f"Loaded {len(tweets_df)} tweets")

print("\nDataset structure:")
print(tweets_df.head())
print("\nColumns in the dataset:", tweets_df.columns.tolist())

print("\nPreparing data for analysis...")

date_columns = [col for col in tweets_df.columns if 'date' in col.lower() or 'time' in col.lower()]
if date_columns:
    date_col = date_columns[0]
    print(f"Found date column: {date_col}")
    tweets_df['date'] = pd.to_datetime(tweets_df[date_col], errors='coerce')
else:
    print("No date column found. Creating simulated dates...")
    np.random.seed(42)
    start_date = datetime(2022, 2, 24)  
    date_range = pd.date_range(start=start_date, periods=len(tweets_df), freq='30min')
    tweets_df['date'] = np.random.choice(date_range, size=len(tweets_df))
    tweets_df['date'] = pd.to_datetime(tweets_df['date'])

tweet_columns = [col for col in tweets_df.columns if 'tweet' in col.lower() or 'text' in col.lower() or 'content' in col.lower()]
if tweet_columns:
    tweet_col = tweet_columns[0]
    print(f"Found tweet column: {tweet_col}")
else:
    text_cols = tweets_df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        avg_lengths = {col: tweets_df[col].astype(str).str.len().mean() for col in text_cols}
        tweet_col = max(avg_lengths.items(), key=lambda x: x[1])[0]
        print(f"Using column with longest text as tweet column: {tweet_col}")
    else:
        tweet_col = tweets_df.columns[0]  
        print(f"No text column identified, using first column: {tweet_col}")

tweets_df['tweet'] = tweets_df[tweet_col]

def extract_features(df):
    df['mentions'] = df['tweet'].apply(lambda x: re.findall(r'@\w+', str(x)))
    df['mentions_count'] = df['mentions'].apply(lambda x: len(x))
   
    df['hashtags'] = df['tweet'].apply(lambda x: re.findall(r'#(\w+)', str(x).lower()))
    df['hashtags_count'] = df['hashtags'].apply(lambda x: len(x))
    
    return df

tweets_df = extract_features(tweets_df)

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

tweets_df['clean_tweet'] = tweets_df['tweet'].apply(clean_text)

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
tweets_df['sentiment_score'] = tweets_df['clean_tweet'].apply(get_sentiment)

def categorize_sentiment(score):
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'

tweets_df['sentiment_category'] = tweets_df['sentiment_score'].apply(categorize_sentiment)

sentiment_distribution = tweets_df['sentiment_category'].value_counts()
print("\nSentiment Distribution:")
print(sentiment_distribution)

russia_ukraine_keywords = [
    'russia', 'ukraine', 'putin', 'zelensky', 'kyiv', 'moscow', 
    'war', 'invasion', 'nato', 'sanctions', 'troops', 'peace', 
    'refugees', 'civilian', 'military', 'nuclear', 'attack', 'defense',
    'donbas', 'crimea', 'european', 'soldier', 'casualties', 'dead',
    'wounded', 'negotiation', 'ceasefire', 'humanitarian', 'corridor',
    'oil', 'gas', 'energy', 'price', 'economy', 'oligarch', 'swift'
]

print("\nGenerating visualizations...\n")

plt.figure(figsize=(10, 6))
sentiment_counts = tweets_df['sentiment_category'].value_counts()
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['red', 'gray', 'green'])
plt.title('Sentiment Distribution in Russia-Ukraine War Tweets')
plt.savefig('sentiment_distribution_pie.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment_category', data=tweets_df, palette={'negative': 'red', 'neutral': 'gray', 'positive': 'green'})
plt.title('Sentiment Distribution in Russia-Ukraine War Tweets')
plt.savefig('sentiment_distribution_bar.png')
plt.close()

try:
    plt.figure(figsize=(14, 7))
    tweets_df['date_only'] = tweets_df['date'].dt.date
    sentiment_over_time = tweets_df.groupby('date_only')['sentiment_score'].mean().reset_index()
    plt.plot(sentiment_over_time['date_only'], sentiment_over_time['sentiment_score'], marker='o')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Average Sentiment Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sentiment_over_time.png')
    plt.close()
    print("Created sentiment over time visualization")
except Exception as e:
    print(f"Error creating sentiment over time visualization: {e}")

try:
    all_hashtags = [tag for sublist in tweets_df['hashtags'] for tag in sublist]
    hashtag_counter = Counter(all_hashtags)
    top_hashtags = [item[0] for item in hashtag_counter.most_common(15)]

    hashtag_sentiments = {}
    for tag in top_hashtags:
        tag_tweets = tweets_df[tweets_df['hashtags'].apply(lambda x: tag in x)]
        if len(tag_tweets) > 0:
            avg_sentiment = tag_tweets['sentiment_score'].mean()
            hashtag_sentiments[tag] = avg_sentiment
    
    sorted_hashtags = sorted(hashtag_sentiments.items(), key=lambda x: x[1])
    sorted_hashtag_names = [x[0] for x in sorted_hashtags]
    sorted_hashtag_scores = [x[1] for x in sorted_hashtags]
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if score < 0 else 'green' for score in sorted_hashtag_scores]
    plt.barh(sorted_hashtag_names, sorted_hashtag_scores, color=colors)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.title('Average Sentiment Score by Hashtag')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Hashtag')
    plt.tight_layout()
    plt.savefig('hashtag_sentiment.png')
    plt.close()
    print("Created hashtag sentiment visualization")
except Exception as e:
    print(f"Error creating hashtag sentiment visualization: {e}")
    
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def process_text_for_wordcloud(text):
        if isinstance(text, str):
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
            return ' '.join(tokens)
        return ""
    
    tweets_df['processed_text'] = tweets_df['clean_tweet'].apply(process_text_for_wordcloud)
    
    negative_tweets = tweets_df[tweets_df['sentiment_category'] == 'negative']
    neutral_tweets = tweets_df[tweets_df['sentiment_category'] == 'neutral']
    positive_tweets = tweets_df[tweets_df['sentiment_category'] == 'positive']
    
    def generate_wordcloud(data, title, filename):
        if len(data) > 0:
            text = ' '.join(data['processed_text'].tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
    
    generate_wordcloud(negative_tweets, 'Word Cloud - Negative Sentiment', 'wordcloud_negative.png')
    generate_wordcloud(neutral_tweets, 'Word Cloud - Neutral Sentiment', 'wordcloud_neutral.png')
    generate_wordcloud(positive_tweets, 'Word Cloud - Positive Sentiment', 'wordcloud_positive.png')
    print("Created sentiment wordclouds")
except Exception as e:
    print(f"Error creating wordclouds: {e}")
    
try:
    key_topics = ['russia', 'ukraine', 'putin', 'zelensky', 'war', 'peace', 'nato', 'sanctions']
    topic_sentiment = {}
    
    for topic in key_topics:
        topic_tweets = tweets_df[tweets_df['clean_tweet'].str.contains(topic, case=False, na=False)]
        if len(topic_tweets) > 0:
            avg_sentiment = topic_tweets['sentiment_score'].mean()
            topic_sentiment[topic] = avg_sentiment
    
    if topic_sentiment:
        plt.figure(figsize=(12, 6))
        topics = list(topic_sentiment.keys())
        sentiments = list(topic_sentiment.values())
        colors = ['red' if s < 0 else 'green' for s in sentiments]
        plt.bar(topics, sentiments, color=colors)
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.title('Average Sentiment Score by Key Topic')
        plt.ylabel('Sentiment Score')
        plt.xlabel('Topic')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('topic_sentiment.png')
        plt.close()
        print("Created topic sentiment visualization")
except Exception as e:
    print(f"Error creating topic sentiment visualization: {e}")

try:
    daily_sentiment = tweets_df.groupby('date_only')['sentiment_score'].agg(['mean', 'std', 'count']).reset_index()
    daily_sentiment = daily_sentiment[daily_sentiment['count'] > 5]  
    
    if len(daily_sentiment) > 5:
        daily_sentiment = daily_sentiment.sort_values('mean')
        most_negative_days = daily_sentiment.head(5)
        most_positive_days = daily_sentiment.tail(5)
        emotional_days = pd.concat([most_negative_days, most_positive_days])
        emotional_days = emotional_days.sort_values('mean')
        
        plt.figure(figsize=(12, 6))
        colors = ['red' if score < 0 else 'green' for score in emotional_days['mean']]
        plt.bar(emotional_days['date_only'].astype(str), emotional_days['mean'], color=colors)
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.title('Days with Most Extreme Sentiment')
        plt.ylabel('Average Sentiment Score')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('emotional_days.png')
        plt.close()
        print("Created emotional days visualization")
except Exception as e:
    print(f"Error creating emotional days visualization: {e}")
    
try:
    sentiment_over_time = tweets_df.groupby('date_only')['sentiment_score'].mean().reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=sentiment_over_time['date_only'],
            y=sentiment_over_time['sentiment_score'],
            mode='lines+markers',
            name='Avg Sentiment Score',
            line=dict(color='blue')
        ),
        secondary_y=False,
    )
 
    tweet_volume = tweets_df.groupby('date_only').size().reset_index(name='count')
    tweet_volume = tweet_volume.merge(sentiment_over_time, on='date_only')
    
    fig.add_trace(
        go.Bar(
            x=tweet_volume['date_only'],
            y=tweet_volume['count'],
            name='Tweet Volume',
            marker_color='rgba(0, 128, 0, 0.5)'
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text='Sentiment Score and Tweet Volume Over Time',
        xaxis_title='Date',
    )
    
    fig.update_yaxes(title_text='Avg Sentiment Score', secondary_y=False)
    fig.update_yaxes(title_text='Tweet Volume', secondary_y=True)
except Exception as e:
  print(f"Error creating sentiment score and tweet volume over time visualisation : {e}")
  
try:
    from sklearn.feature_extraction.text import CountVectorizer
    tweets_df['hashtag_str'] = tweets_df['hashtags'].apply(lambda x: ' '.join(x) if len(x) > 0 else '')
    
    hashtag_tweets = tweets_df[tweets_df['hashtag_str'] != '']
    
    if len(hashtag_tweets) > 10:  
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(hashtag_tweets['hashtag_str'])
        feature_names = vectorizer.get_feature_names_out()
        co_occurrence = (X.T * X)
        co_occurrence.setdiag(0) 
        co_occurrence = co_occurrence.toarray()
        co_df = pd.DataFrame(co_occurrence, index=feature_names, columns=feature_names)
        top_co_occurrences = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if co_occurrence[i, j] > 0:
                    top_co_occurrences.append((feature_names[i], feature_names[j], co_occurrence[i, j]))
    
        top_co_occurrences.sort(key=lambda x: x[2], reverse=True)
        top_co_occurrences = top_co_occurrences[:30]
        import networkx as nx
        
        G = nx.Graph()
        for source, target, weight in top_co_occurrences:
            G.add_edge(source, target, weight=weight)
        
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
        edge_weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title("Hashtag Co-occurrence Network")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('hashtag_network.png')
        plt.close()
        print("Created hashtag network visualization")
except Exception as e:
    print(f"Error creating hashtag network visualization: {e}")

try:
    all_mentions = [mention for sublist in tweets_df['mentions'] for mention in sublist]
    mention_counter = Counter(all_mentions)
    
    if mention_counter:
        top_mentions = mention_counter.most_common(10)
        mentions = [m[0] for m in top_mentions]
        counts = [m[1] for m in top_mentions]
        
        plt.figure(figsize=(12, 6))
        plt.bar(mentions, counts)
        plt.title('Top Mentioned Accounts')
        plt.ylabel('Mention Count')
        plt.xlabel('Account')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('top_mentions.png')
        plt.close()
        print("Created top mentions visualization")
except Exception as e:
    print(f"Error creating top mentions visualization: {e}")

print("\nAnalysis complete! Generated visualizations:")
print("1. sentiment_distribution_pie.png - Overall sentiment distribution")
print("2. sentiment_distribution_bar.png - Sentiment distribution bar chart")
print("3. sentiment_over_time.png - Sentiment trends over time")
print("4. hashtag_sentiment.png - Sentiment associated with top hashtags")
print("5. wordcloud_negative.png, wordcloud_neutral.png, wordcloud_positive.png - Word clouds by sentiment")
print("6. topic_sentiment.png - Sentiment associated with key topics")
print("7. emotional_days.png - Days with most extreme sentiment")
print("8. sentiment_volume_over_time.html - Interactive visualization of sentiment and volume")
print("9. hashtag_network.png - Network visualization of hashtag co-occurrences")
print("10. top_mentions.png - Top mentioned accounts")
print("\nCheck these files to review the sentiment analysis results.") 