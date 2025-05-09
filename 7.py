# #part 1
# # Import necessary libraries
# import nltk
# from nltk.tokenize import word_tokenize
# import string
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords, wordnet
# from nltk import pos_tag
# from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# # Download required NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# # Sample document
# sample_text = """
# Artificial Intelligence is transforming industries rapidly. Machines can now understand human language, 
# learn from data, and make decisions. This evolution brings both excitement and challenges.
# """

# # 1. Tokenization
# tokens = word_tokenize(sample_text)
# print("Tokens:", tokens)

# # 2. POS Tagging
# pos_tags = pos_tag(tokens)
# print("\nPOS Tags:", pos_tags)

# # 3. Stop Words Removal
# stop_words = set(stopwords.words('english'))
# filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
# print("\nFiltered Tokens (No Stopwords):", filtered_tokens)

# # 4. Stemming
# stemmer = PorterStemmer()
# stemmed = [stemmer.stem(word) for word in filtered_tokens]
# print("\nStemmed Words:", stemmed)

# # 5. Lemmatization (requires POS mapping)
# lemmatizer = WordNetLemmatizer()

# # Function to convert nltk POS tag to wordnet POS tag
# def get_wordnet_pos(tag):
#     if tag.startswith('J'):
#         return wordnet.ADJ
#     elif tag.startswith('V'):
#         return wordnet.VERB
#     elif tag.startswith('N'):
#         return wordnet.NOUN
#     elif tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN  # Default to noun

# lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tag(filtered_tokens)]
# print("\nLemmatized Words:", lemmatized)


##part 2
# Sample corpus with multiple documents
documents = [
    "Artificial Intelligence is transforming industries rapidly.",
    "Machines can now understand human language.",
    "Learn from data and make decisions.",
    "This evolution brings both excitement and challenges."
]

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names
features = vectorizer.get_feature_names_out()

# Convert TF-IDF matrix to DataFrame for better display
import pandas as pd
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=features)
print("\nTF-IDF Matrix:")
print(df_tfidf.round(2))

