# Russia-Ukraine War Tweet Sentiment Analysis

This project analyzes sentiment in tweets related to the Russia-Ukraine conflict, providing visualizations to understand public opinion trends.

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Make sure the tweet dataset is in the root directory:
   - 30K Tweets with russiaukrainewar hashtag.csv

## Running the Analysis

1. Execute the main analysis script to generate visualizations:
   ```
   python russia_ukraine_sentiment_analysis.py
   ```

2. Launch the interactive Streamlit dashboard:
   ```
   streamlit run streamlit_dashboard.py
   ```

## Dashboard Features

The Streamlit dashboard offers:
- Interactive visualization selection
- Proper sentiment color coding (green=positive, gray=neutral, red=negative)
- Detailed descriptions of each chart
- Analysis summary view

## Visualizations Available

1. **Overall Sentiment Distribution (Pie)**: Distribution of positive, negative, and neutral tweets
2. **Sentiment Distribution Bar Chart**: Count of tweets by sentiment
3. **Sentiment Over Time**: Tracking sentiment changes throughout the conflict
4. **Hashtag Sentiment Analysis**: Which hashtags associate with positive or negative sentiment
5. **Word Clouds**: Most common words in positive, neutral, and negative tweets
6. **Key Topics Sentiment**: Sentiment scores for Russia, Ukraine, NATO, etc.
7. **Most Emotional Days**: Days with extreme sentiment scores
8. **Hashtag Co-occurrence Network**: Which hashtags appear together
9. **Top Mentioned Accounts**: Most referenced accounts in tweets


