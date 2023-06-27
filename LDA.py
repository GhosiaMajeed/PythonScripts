import pandas as pd
import numpy as np
import re
import string
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Load the CSV file
df = pd.read_csv(r'C:\Users\Ghosia\job_listings.csv')
df=df.astype(str)
more_stopwords = ['the','and','to','in','of','need','want','needed','looking','create','needs','create','help','using','required']
existing_stopwords = set(stopwords.words('english'))
stop_words = existing_stopwords.union(more_stopwords)
# Preprocess the text column
def preprocess_text(text):
    text = text.lower() # convert to lowercase
    #stop_words.update(more_stopwords)
    text = re.sub('\[.*?\]', '', text) # remove square brackets
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    # Join the filtered words back into a sentence
    text = ' '.join(filtered_words)
    text = re.sub('\w*\d\w*', '', text) # remove words containing digits
    return text

# Apply the preprocessing function to the text column
df['text_processed'] = df['Title'].apply(preprocess_text)

# Create a dictionary from the preprocessed text
dictionary = corpora.Dictionary(df['text_processed'].apply(lambda x: x.split()))

# Create a corpus from the dictionary and preprocessed text
corpus = [dictionary.doc2bow(text.split()) for text in df['text_processed']]

# Train the LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=30, passes=250)

# Calculate the topic frequency in percentage
topic_freq = np.zeros(lda_model.num_topics)
for doc_topics in lda_model.get_document_topics(corpus):
    for topic, freq in doc_topics:
        topic_freq[topic] += freq

topic_freq_pct = topic_freq / topic_freq.sum() * 100

# Print the topic frequency in percentage
for i, freq in enumerate(topic_freq_pct):
    print("Topic {}: {:.2f}%".format(i, freq))

""" for topic_id, topic in lda_model.print_topics():
    print("Topic {}: {}".format(topic_id, topic)) """

for topic_id in range(lda_model.num_topics):
    words = lda_model.show_topic(topic_id, topn=10)
    topic = f"Topic {topic_id}: "
    keywords = ", ".join([word[0] for word in words])
    print(topic + keywords)

# Visualize the topics using a bar chart
topics = lda_model.show_topics(num_topics=30, num_words=7, formatted=False)

for i in range(len(topics)):
    top_words = [w[0] for w in topics[i][1]]
    weights = [w[1] for w in topics[i][1]]
    plt.barh(top_words, weights)
    plt.title(f'Topic {i+1}')
    plt.show()


# Get the topics and their frequencies

topic_freqs = [sum([w[1] for w in t[1]]) for t in topics]
print(topic_freqs)
# Plot the topics and their frequencies as a horizontal bar chart
plt.barh([f'Topic {i+1}' for i in range(len(topics))], topic_freqs)
plt.ylabel('Topic')
plt.xlabel('Frequency')
plt.title('Top Topics and Their Frequencies')
plt.show()

