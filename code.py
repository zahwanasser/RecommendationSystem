
# coding: utf-8

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


# reading file
book_description = pd.read_csv(r"C:\Users\D\Desktop\math project\Recommendation-systems-using-python-master\Recommendation-systems-using-python-master\description.csv", encoding = 'latin-1')


# In[3]:


# checking if we have the right data
book_description.head()


# In[4]:


# removing the stop words
books_tfidf = TfidfVectorizer(stop_words='english')
# filling the missing values with empty string
book_description['description'] = book_description['description'].fillna('')
# computing TF-IDF matrix required for calculating cosine similarity
book_description_matrix = books_tfidf.fit_transform(book_description['description'])


# In[5]:


# Let's check the shape of computed matrix
book_description_matrix.shape


# In[6]:


# Let's check the shape of computed matrix
book_description_matrix.shape


# In[7]:


# computing cosine similarity matrix using linear_kernal of sklearn
cosine_similarity = linear_kernel(book_description_matrix, book_description_matrix)


# In[8]:


print(cosine_similarity)


# In[9]:


#printing the tf-idf weights for each relevant word in the description of each book in the dataset
print(book_description_matrix) 


# In[10]:


#seperating the book names column from the dataframe
df=book_description
df1=df.loc[:,"name":"name"]

print(df1)


# In[11]:


# Function to get the most similar books
def recommend(index, cosine_sim=cosine_similarity):
    # Get the pairwsie similarity scores of all books compared to that book, 
    # sorting them and getting top 5
    id=index
    similarity_scores = list(enumerate(cosine_sim[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]

    # Get the books index
    books_index = [i[0] for i in similarity_scores]

    # Return the top 5 most similar books using integer-location based indexing (iloc)
    return book_description['name'].iloc[books_index]


# In[12]:


# getting the input of the book which i want to have similar recommendations to it and matching it to its index in the dataset
df2=df1[df1['name' ].str.match(input().title(), na=False)]
print(df2)
#getting the index of the book
df3=df2.index.astype(int)
x=0
for i in range(142):
    x=x+1
# getting recommendation for book 
print('you may also like to read: ')
recommend(int(df3.values[0]))

