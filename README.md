# Code Alpha_sentimental-Analysis
Sentiment analysis, also called opinion mining, uses Natural Language Processing (NLP) to automatically determine the emotional tone (positive, negative, or neutral) in text, extracting subjective opinions from customer reviews, social media, surveys, etc., to understand public feeling about a product, brand, or topic. .

import pandas as pd

# Load the CSV file from the zip archive into a pandas DataFrame
df = pd.read_csv('/content/kalki_movie_reviews.csv.zip')

# Display the first 5 rows of the DataFrame
print("First 5 rows of the DataFrame:")
print(df.head())

print("DataFrame Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics for Numerical Columns:")
print(df.describe())

print("\nUnique values and counts for 'Ratings' column:")
print(df['Ratings'].value_counts().sort_index())

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def get_sentiment_vader(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the function to the 'Comments' column
df['Sentiment'] = df['Comments'].apply(get_sentiment_vader)

# Display the count of each sentiment label
print("Sentiment Distribution:")
print(df['Sentiment'].value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for better aesthetics
sns.set_style('whitegrid')

# Create a bar plot of the sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=df, palette='viridis', order=['Positive', 'Neutral', 'Negative'])
plt.title('Distribution of Movie Review Sentiments', fontsize=16)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.show()

print("Sentiment distribution visualization generated.")

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for better aesthetics
sns.set_style('whitegrid')

# Create a bar plot of the sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', hue='Sentiment', data=df, palette='viridis', order=['Positive', 'Neutral', 'Negative'], legend=False)
plt.title('Distribution of Movie Review Sentiments', fontsize=16)
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.show()

print("Sentiment distribution visualization generated.")
