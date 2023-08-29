import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load sample text
sample_text = """
    Text analysis involves processing and understanding text data in various ways. It includes tasks such as tokenization, 
    part-of-speech tagging, sentiment analysis, named entity recognition, and more. This code demonstrates how to perform 
    these tasks using popular Python libraries.
"""

# Tokenization and stop words removal
tokens = word_tokenize(sample_text)
filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]

# Frequency distribution
fdist = FreqDist(filtered_tokens)
print("Top 10 most common words:")
print(fdist.most_common(10))

# Sentiment analysis
sid = SentimentIntensityAnalyzer()
sentiment_scores = sid.polarity_scores(sample_text)
print("\nSentiment analysis scores:")
print("Positive:", sentiment_scores['pos'])
print("Neutral:", sentiment_scores['neu'])
print("Negative:", sentiment_scores['neg'])

# Named Entity Recognition using spaCy
doc = nlp(sample_text)
print("\nNamed Entities:")
for ent in doc.ents:
    print(ent.text, "-", ent.label_)

# TF-IDF Vectorization
corpus = [sample_text]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print("\nTF-IDF Features:")
print(vectorizer.get_feature_names_out())

# K-Means Clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(X)
print("\nCluster centers:")
print(kmeans.cluster_centers_)

# Plotting K-Means Clusters
plt.scatter(X.toarray()[:, 0], X.toarray()[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='black')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
