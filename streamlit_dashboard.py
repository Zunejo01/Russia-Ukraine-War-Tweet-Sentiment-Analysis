import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.colors as mcolors
import io
from textblob import TextBlob
import random

st.set_page_config(
    page_title="Russia-Ukraine War Tweet Sentiment Analysis",
    layout="wide"
)

st.title("Russia-Ukraine War Tweet Sentiment Analysis")
st.markdown("Interactive dashboard for analyzing sentiment in tweets related to the Russia-Ukraine conflict.")

def get_image_path(filename):
    return filename if os.path.exists(filename) else None

@st.cache_data
def load_tweet_data():
    if os.path.exists('30K Tweets with russiaukrainewar hashtag.csv'):
        try:
            df = pd.read_csv('30K Tweets with russiaukrainewar hashtag.csv')
            tweet_cols = [col for col in df.columns if 'tweet' in col.lower() or 'text' in col.lower() or 'content' in col.lower()]
            if not tweet_cols:
                st.error("No tweet column found in the dataset")
                return None
                
            tweet_col = tweet_cols[0]
            
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            date_col = date_cols[0] if date_cols else None
            
            if date_col:
                df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.rename(columns={tweet_col: 'tweet'})
        
            if 'sentiment_score' not in df.columns:
                df['sentiment_score'] = df['tweet'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
                
                def categorize_sentiment(score):
                    if score > 0.1:
                        return 'positive'
                    elif score < -0.1:
                        return 'negative'
                    else:
                        return 'neutral'
                
                df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
            
            return df
        except Exception as e:
            st.error(f"Error loading tweet data: {e}")
            return None
    else:
        st.error("Tweet dataset not found. Please ensure '30K Tweets with russiaukrainewar hashtag.csv' is in the root directory.")
        return None
        
def analyze_user_text(text):
    if not text.strip():
        return None, None, None
        
    analysis = TextBlob(text)
    score = analysis.sentiment.polarity
    
    if score > 0.1:
        category = 'positive'
        color = 'green'
    elif score < -0.1:
        category = 'negative'
        color = 'red'
    else:
        category = 'neutral'
        color = 'grey'
        
    return score, category, color
    
viz_paths = {
    'sentiment_pie': get_image_path('sentiment_distribution_pie.png'),
    'sentiment_bar': get_image_path('sentiment_distribution_bar.png'),
    'sentiment_time': get_image_path('sentiment_over_time.png'),
    'hashtag_sentiment': get_image_path('hashtag_sentiment.png'),
    'topic_sentiment': get_image_path('topic_sentiment.png'),
    'emotional_days': get_image_path('emotional_days.png'),
    'hashtag_network': get_image_path('hashtag_network.png'),
    'top_mentions': get_image_path('top_mentions.png'),
    'wordcloud_positive': get_image_path('wordcloud_positive.png'),
    'wordcloud_neutral': get_image_path('wordcloud_neutral.png'),
    'wordcloud_negative': get_image_path('wordcloud_negative.png'),
}

try:
    with open('analysis_summary.md', 'r', encoding='utf-8') as f:
        summary_content = f.read()
except Exception as e:
    summary_content = "Analysis summary not available."

viz_options = {
    'sentiment_pie': 'Overall Sentiment Distribution (Pie)',
    'sentiment_bar': 'Overall Sentiment Distribution (Bar)',
    'sentiment_time': 'Sentiment Over Time',
    'hashtag_sentiment': 'Sentiment by Hashtag',
    'topic_sentiment': 'Sentiment by Topic',
    'emotional_days': 'Most Emotional Days',
    'hashtag_network': 'Hashtag Co-occurrence Network',
    'top_mentions': 'Top Mentioned Accounts',
    'wordcloud_positive': 'Word Cloud - Positive Sentiment',
    'wordcloud_neutral': 'Word Cloud - Neutral Sentiment',
    'wordcloud_negative': 'Word Cloud - Negative Sentiment',
    'summary': 'Analysis Summary',
    
    'search_filter': 'Search & Filter Tweets',
    'sentiment_analyzer': 'Analyze Your Own Text',
    'topic_explorer': 'Topic Explorer',
    'accuracy_checker': 'Sentiment Classifier Accuracy Checker'
}

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Section",
    options=["Visualizations", "Interactive Features"]
)
descriptions = {
    'sentiment_pie': "Pie chart showing the overall distribution of sentiment in tweets about the Russia-Ukraine war. Categories include positive (green), negative (red), and neutral (grey) sentiment.",
    'sentiment_bar': "Bar chart showing the count of tweets in each sentiment category (positive, negative, neutral).",
    'sentiment_time': "Line chart showing how the average sentiment score changes over time. Values above 0 indicate positive sentiment, while values below 0 indicate negative sentiment.",
    'hashtag_sentiment': "Bar chart showing the average sentiment associated with the most common hashtags in the dataset.",
    'topic_sentiment': "Bar chart showing the average sentiment score associated with key topics like 'Russia', 'Ukraine', 'NATO', etc.",
    'emotional_days': "Bar chart showing the days with the most extreme sentiment scores, both positive and negative.",
    'hashtag_network': "Network visualization showing which hashtags commonly appear together in tweets, with line thickness indicating frequency of co-occurrence.",
    'top_mentions': "Bar chart showing the most frequently mentioned accounts in tweets about the Russia-Ukraine war.",
    'wordcloud_positive': "Word cloud showing the most common words in tweets with positive sentiment. Larger words appear more frequently.",
    'wordcloud_neutral': "Word cloud showing the most common words in tweets with neutral sentiment. Larger words appear more frequently.",
    'wordcloud_negative': "Word cloud showing the most common words in tweets with negative sentiment. Larger words appear more frequently.",
    'summary': "Summary of key findings from the sentiment analysis.",
   
    'search_filter': "Search for specific keywords in tweets and filter by sentiment or date.",
    'sentiment_analyzer': "Analyze the sentiment of your own text using the same model applied to the tweet dataset.",
    'topic_explorer': "Explore tweets and sentiment around specific topics or hashtags in the dataset.",
    'accuracy_checker': "Evaluate the accuracy of our sentiment classifier by reviewing random tweets."
}

def generate_corrected_sentiment_pie():

    try:

        if os.path.exists('sentiment_distribution_pie.png'):
            sentiment_counts = {
                'positive': 30,
                'neutral': 45,
                'negative': 25
            }
            try:
                df = load_tweet_data()
                if df is not None and 'sentiment_category' in df.columns:
                    sentiment_counts = df['sentiment_category'].value_counts().to_dict()
                    
                    for category in ['positive', 'negative', 'neutral']:
                        if category not in sentiment_counts:
                            sentiment_counts[category] = 0
            except Exception as e:
                st.sidebar.warning(f"Using estimated sentiment distribution: {e}")
        
        fig, ax = plt.figure(figsize=(10, 6)), plt.subplot()
        labels = ['positive', 'neutral', 'negative']
        counts = [sentiment_counts.get(label, 0) for label in labels]
        colors = ['green', 'grey', 'red']  # This matches labels in order
        
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=labels, 
            autopct='%1.1f%%', 
            colors=colors,
            startangle=90
        )
    
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
            
        ax.axis('equal')  
        plt.title('Sentiment Distribution in Russia-Ukraine War Tweets', fontsize=16)
        ax.legend(wedges, [f"{label.capitalize()} ({count})" for label, count in zip(labels, counts)], 
                 title="Sentiment", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        return fig
    except Exception as e:
        st.error(f"Error generating pie chart: {e}")
        return None

if section == "Visualizations":
    available_viz = [name for name, path in viz_paths.items() if path]
    if not available_viz:
        st.warning("No visualization files found. Please run the sentiment analysis script first.")
    
    st.sidebar.title("Visualization Options")
    viz_list = list(filter(lambda x: x in viz_options and (x in viz_paths or x == 'summary'), viz_options.keys()))
    selected_viz = st.sidebar.selectbox(
        "Select Visualization",
        options=viz_list,
        format_func=lambda x: viz_options.get(x, x),
        index=0 if viz_paths.get('sentiment_pie') else len(viz_list) - 1  # Default to summary if no files
    )

    st.subheader(viz_options.get(selected_viz, selected_viz))

    if selected_viz == 'summary':
        st.markdown(summary_content)
    elif selected_viz == 'sentiment_pie':
        fig = generate_corrected_sentiment_pie()
        if fig:
            st.pyplot(fig)
        else:
            st.error("Could not generate the sentiment pie chart with correct colors.")
        
        st.info(descriptions['sentiment_pie'])
    else:
        path = viz_paths.get(selected_viz)
        if path and os.path.exists(path):
            try:
                img = Image.open(path)
                st.image(img, use_container_width=True)
                
                if selected_viz in descriptions:
                    st.info(descriptions[selected_viz])
            except Exception as e:
                st.error(f"Error displaying image: {e}")
        else:
            st.warning(f"Visualization file not found: {path}")
            st.info("Please run the sentiment analysis script first to generate the visualizations.")

else:  
    st.sidebar.title("Interactive Features")
    feature = st.sidebar.selectbox(
        "Select Feature",
        options=['search_filter', 'sentiment_analyzer', 'topic_explorer', 'accuracy_checker'],
        format_func=lambda x: viz_options.get(x, x)
    )

    df = load_tweet_data()
    
    if df is None:
        st.error("Cannot load tweet data. Please ensure the CSV file exists and is properly formatted.")
    else:
        if feature == 'search_filter':
            st.subheader("🔍 Search & Filter Tweets")
            st.info(descriptions['search_filter'])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                search_term = st.text_input("Search for keywords in tweets:", "")
                
            with col2:
                sentiment_filter = st.selectbox(
                    "Filter by sentiment:",
                    options=["All", "Positive", "Neutral", "Negative"]
                )
    
            if 'date' in df.columns:
                date_min, date_max = df['date'].min(), df['date'].max()
                date_range = st.slider(
                    "Select date range:",
                    min_value=date_min.date(),
                    max_value=date_max.date(),
                    value=(date_min.date(), date_max.date())
                )
                
            
                mask_date = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
                df_filtered = df[mask_date]
            else:
                df_filtered = df.copy()
            
            if sentiment_filter != "All":
                df_filtered = df_filtered[df_filtered['sentiment_category'].str.lower() == sentiment_filter.lower()]
        
            if search_term:
                df_filtered = df_filtered[df_filtered['tweet'].str.contains(search_term, case=False, na=False)]
            
            st.write(f"Found {len(df_filtered)} matching tweets")
            
            if not df_filtered.empty:
                st.subheader("Sample Tweets")
                sample_size = min(10, len(df_filtered))
                
                for i, row in df_filtered.head(sample_size).iterrows():
                    sentiment = row['sentiment_category']
                    score = row['sentiment_score']
                    
                    if sentiment == 'positive':
                        color = 'green'
                    elif sentiment == 'negative':
                        color = 'red'
                    else:
                        color = 'grey'
                    
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px;">
                        <p>{row['tweet']}</p>
                        <p style="color:{color}; margin-bottom:0;">
                            <strong>Sentiment:</strong> {sentiment.capitalize()} ({score:.2f})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No matching tweets found. Try different search criteria.")

        elif feature == 'sentiment_analyzer':
            st.subheader("✍️ Analyze Your Own Text")
            st.info(descriptions['sentiment_analyzer'])
            
            user_text = st.text_area("Enter your text to analyze sentiment:", 
                                   height=150,
                                   placeholder="E.g., The war is devastating for civilian populations")
            
            analyze_button = st.button("Analyze Sentiment")
            
            if analyze_button and user_text:
                score, category, color = analyze_user_text(user_text)
                
                if score is not None:
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:5px; padding:20px; margin-top:20px;">
                        <h3 style="color:{color};">Sentiment: {category.capitalize()}</h3>
                        <p style="font-size:18px;">Score: {score:.2f}</p>
                        <p>
                            <strong>Interpretation:</strong><br>
                            Scores range from -1.0 (very negative) to 1.0 (very positive).<br>
                            Scores between -0.1 and 0.1 are considered neutral.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if df is not None:
                        avg_score = df['sentiment_score'].mean()
                        st.write(f"For comparison, the average sentiment score in the dataset is {avg_score:.2f}")
                        
                        if len(df) > 0:
                            similar_range = 0.1
                            similar_tweets = df[(df['sentiment_score'] >= score - similar_range) & 
                                              (df['sentiment_score'] <= score + similar_range)]
                            
                            if len(similar_tweets) > 0:
                                st.subheader("Tweets with similar sentiment:")
                                for _, row in similar_tweets.head(3).iterrows():
                                    st.write(f"• \"{row['tweet']}\" (Score: {row['sentiment_score']:.2f})")
            elif analyze_button:
                st.warning("Please enter some text to analyze.")

        elif feature == 'topic_explorer':
            st.subheader("🔍 Topic Explorer")
            st.info(descriptions['topic_explorer'])
        
            topics = [
                'Russia', 'Ukraine', 'Putin', 'Zelensky', 'NATO', 'War', 'Peace',
                'Sanctions', 'Refugees', 'Humanitarian', 'Nuclear', 'Kyiv', 'Moscow'
            ]
            
            if 'tweet' in df.columns:
                import re
                hashtags = []
                for tweet in df['tweet'].dropna():
                    tags = re.findall(r'#(\w+)', tweet)
                    hashtags.extend(tags)
                
                from collections import Counter
                hashtag_counts = Counter(hashtags)
                top_hashtags = [tag for tag, count in hashtag_counts.most_common(20) 
                               if tag.lower() != 'russiaukrainewar']  # Exclude the main hashtag
                
                all_topics = topics + [f'#{tag}' for tag in top_hashtags]
            else:
                all_topics = topics
            
            selected_topic = st.selectbox(
                "Select a topic or hashtag to explore:",
                options=all_topics
            )
            
            if selected_topic:
                search_term = selected_topic[1:] if selected_topic.startswith('#') else selected_topic
                
                topic_tweets = df[df['tweet'].str.contains(search_term, case=False, na=False)]
                
                if len(topic_tweets) > 0:
                    topic_sentiment = topic_tweets['sentiment_category'].value_counts()
                    avg_score = topic_tweets['sentiment_score'].mean()
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader(f"Sentiment for \"{selected_topic}\"")
                        st.write(f"Average sentiment score: {avg_score:.2f}")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        colors = {'positive': 'green', 'neutral': 'grey', 'negative': 'red'}
                    
                        for cat in ['positive', 'neutral', 'negative']:
                            if cat not in topic_sentiment:
                                topic_sentiment[cat] = 0
                        
                        ax.bar(topic_sentiment.index, topic_sentiment.values, 
                              color=[colors.get(x, 'blue') for x in topic_sentiment.index])
                        ax.set_title(f'Sentiment Distribution for "{selected_topic}"')
                        ax.set_ylabel('Number of Tweets')
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Sample Tweets")
                        
                        for sentiment in ['positive', 'negative', 'neutral']:
                            sentiment_tweets = topic_tweets[topic_tweets['sentiment_category'] == sentiment]
                            
                            if len(sentiment_tweets) > 0:
                                sample = sentiment_tweets.sample(min(1, len(sentiment_tweets)))
                                
                                for _, row in sample.iterrows():
                                    st.markdown(f"""
                                    <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px;">
                                        <p>{row['tweet']}</p>
                                        <p style="color:{colors[sentiment]}; margin-bottom:0;">
                                            <strong>Sentiment:</strong> {sentiment.capitalize()} ({row['sentiment_score']:.2f})
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                else:
                    st.write(f"No tweets found containing \"{selected_topic}\"")

        elif feature == 'accuracy_checker':
            st.subheader("🎯 Sentiment Classifier Accuracy Checker")
            st.info(descriptions['accuracy_checker'])
            
            if 'feedback' not in st.session_state:
                st.session_state.feedback = {}
                
            if 'samples' not in st.session_state:
                st.session_state.samples = df.sample(min(5, len(df))).reset_index()
            
            st.write("Here are 5 random tweets with their predicted sentiment. Do you agree with the classification?")
            for i, row in st.session_state.samples.iterrows():
                tweet = row['tweet']
                predicted = row['sentiment_category']
                score = row['sentiment_score']
                
                if predicted == 'positive':
                    color = 'green'
                elif predicted == 'negative':
                    color = 'red'
                else:
                    color = 'grey'
                
                st.markdown(f"""
                <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px;">
                    <p>{tweet}</p>
                    <p style="color:{color};">
                        <strong>Predicted Sentiment:</strong> {predicted.capitalize()} ({score:.2f})
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
                
                with col1:
                    if st.button("👍 Correct", key=f"correct_{i}"):
                        st.session_state.feedback[i] = "correct"
                        
                with col2:
                    if st.button("👎 Incorrect", key=f"incorrect_{i}"):
                        st.session_state.feedback[i] = "incorrect"
                
                with col3:
                    if st.button("🤷 Unsure", key=f"unsure_{i}"):
                        st.session_state.feedback[i] = "unsure"
                
                feedback = st.session_state.feedback.get(i, None)
                if feedback:
                    if feedback == "correct":
                        st.success("You marked this as correctly classified.")
                    elif feedback == "incorrect":
                        st.error("You marked this as incorrectly classified.")
                    else:
                        st.info("You were unsure about this classification.")
                
                st.markdown("---")
            
            if len(st.session_state.feedback) > 0:
                correct_count = list(st.session_state.feedback.values()).count("correct")
                st.subheader("Feedback Summary")
                st.write(f"You marked {correct_count} out of {len(st.session_state.feedback)} classifications as correct.")
                
                if correct_count / len(st.session_state.feedback) >= 0.6:
                    st.success("The sentiment classifier seems to be performing reasonably well!")
                else:
                    st.warning("The sentiment classifier might need improvement based on your feedback.")
            
            if st.button("Get New Samples"):
                # Reset state
                st.session_state.samples = df.sample(min(5, len(df))).reset_index()
                st.session_state.feedback = {}
                st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Created using Python, Streamlit, and data analysis libraries. Data source: Twitter tweets with #russiaukrainewar hashtag") 
