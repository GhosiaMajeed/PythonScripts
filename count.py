import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(r'C:\Users\Ghosia\job_listings.csv')
df = df.astype(str)

# Concatenate all the skills into a single string separated by commas
skills_string = ','.join(df['Skills'].tolist())

# Split the string into individual skills
all_skills = skills_string.split(',')

# Count the number of unique skills
unique_skills = set(all_skills)
num_unique_skills = len(unique_skills)

# Print the unique skills
print("Unique skills:")
for skill in unique_skills:
    print(skill)

# Print the number of unique skills
print("Number of unique skills:", num_unique_skills)