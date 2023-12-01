import pandas as pd
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Read the CSV file
df = pd.read_csv(r'C:\Users\Ghosia\job_listings.csv')
df = df.astype(str)
""" 
# Extract the skills column as a list
skills_list = [skill.split(', ') for skill in df['Skills'].tolist()]

# Flatten the list of skills
flattened_skills_list = [skill for sublist in skills_list for skill in sublist]

# Create an N-gram model with n=3, excluding custom stop words
ngram_model = CountVectorizer(ngram_range=(3, 3))

# Fit the N-gram model on the flattened skills data
ngram_matrix = ngram_model.fit_transform(flattened_skills_list)

# Get the vocabulary of the N-gram model
vocabulary = ngram_model.vocabulary_

# Invert the vocabulary to get feature names
feature_names = {v: k for k, v in vocabulary.items()}

# Calculate the frequencies of the N-gram combinations
frequencies = ngram_matrix.sum(axis=0).A1

# Create a list to store tuples of N-gram combinations and their frequencies
ngram_list = [(feature_names[index], frequency) for index, frequency in enumerate(frequencies)]

# Sort the list in descending order based on the frequency of each N-gram combination
sorted_ngrams = sorted(ngram_list, key=lambda item: item[1], reverse=True)

# Select the top 10 most frequent N-gram combinations
top_10_ngrams = sorted_ngrams[:10]

# Print the top 10 most frequent N-gram combinations
for ngram, frequency in top_10_ngrams:
    print(ngram, frequency)
 """


# Concatenate all the skills into a single string separated by commas
skills_string = ','.join(df['Skills'].tolist())

# Split the string into individual skills
all_skills = skills_string.split(',')

# Generate triplets of skills
n = 3  # specify the number of skills in each triplet
skill_triplets = list(ngrams(all_skills, n))

# Count the occurrence of each triplet
triplet_counts = Counter(skill_triplets)

# Find the top 20 most frequent triplets
top_20_triplets = triplet_counts.most_common(20)

# Print the top 20 triplets and their counts
print("Top 20 combinations:")
for triplet, count in top_20_triplets:
    print("Combination:", triplet)
    print("Count:", count)
    print("---")