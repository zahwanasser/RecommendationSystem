
# coding: utf-8

# In[1]:


# In[1]:
# importing libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


# In[2]:
# reading file
book_description = pd.read_csv(r"C:\Users\D\Desktop\math project\Recommendation-systems-using-python-master\Recommendation-systems-using-python-master\description.csv", encoding = 'latin-1')


# In[3]:


# In[3]:
# checking if we have the right data
book_description.head()


# In[4]:


# In[4]:
# removing the stop words
books_tfidf = TfidfVectorizer(stop_words='english')
# filling the missing values with empty string
book_description['description'] = book_description['description'].fillna('')
# computing TF-IDF matrix required for calculating cosine similarity
book_description_matrix = books_tfidf.fit_transform(book_description['description'])


# In[5]:


# In[5]:
# Let's check the shape of computed matrix
book_description_matrix.shape


# In[6]:


# In[5]:
# Let's check the shape of computed matrix
book_description_matrix.shape


# In[7]:


# computing cosine similarity matrix using linear_kernal of sklearn
cosine_similarity = linear_kernel(book_description_matrix, book_description_matrix)


# In[8]:


print(book_description_matrix)


# In[9]:


df=book_description
df1=df.loc[:,"name":"name"]

print(df1)


# In[24]:


df2=df1[df1['name' ].str.match(input().title(), na=False)]
print(df2)


# In[25]:


df3=df2.index.astype(int)
x=0
for i in range(142):
    print(int(df3.values[i]))
    x=x+1


# In[20]:


# In[7]:
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


# In[26]:


# In[8]:
# getting recommendation for book at index 2
recommend(int(df3.values[0]))

