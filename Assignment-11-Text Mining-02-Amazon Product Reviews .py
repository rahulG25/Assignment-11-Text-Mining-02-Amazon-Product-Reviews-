#!/usr/bin/env python
# coding: utf-8

# 1) Extract reviews of any product from ecommerce website like amazon, 2) Perform emotion mining

# In[15]:


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import spacy

from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


# Import extracted amazon reviews Dataset (How to Extract amazon reviews - Refer Extract Amazon Reviews using Scrapy.ipynb)
reviews=pd.read_csv('Amazon - reviews.csv')
reviews


# # Text Preprocessing

# In[43]:


reviews=[comment.strip() for comment in reviews] # remove both the leading and the trailing characters
reviews=[comment for comment in reviews if comment] # removes empty strings, because they are considered in Python as False
reviews[0:15]


# In[44]:


# Joining the list into one string/text
reviews_text=' '.join(reviews)
reviews_text


# In[52]:


# Tokenization
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[53]:


from nltk import word_tokenize
text_tokens=word_tokenize(no_punc_text)
print(text_tokens[0:50])


# In[54]:


len(text_tokens)


# In[55]:


# Remove stopwords
from nltk.corpus import stopwords
my_stop_words=stopwords.words('english')

sw_list=['I','The','It','A']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens)


# In[56]:


# Normalize the data
lower_words=[comment.lower() for comment in no_stop_tokens]
print(lower_words)


# In[57]:


# Stemming (Optional)
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]
print(stemmed_tokens)


# In[58]:


# Lemmatization
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))
print(doc)


# In[59]:


lemmas=[token.lemma_ for token in doc]
print(lemmas)


# In[60]:


clean_reviews=' '.join(lemmas)
clean_reviews


# # Feature Extaction
# 1. Using CountVectorizer

# In[61]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
reviewscv=cv.fit_transform(lemmas)


# In[62]:


print(cv.vocabulary_)


# In[64]:


print(cv.get_feature_names()[50:100])


# In[65]:


print(reviewscv.toarray()[50:100])


# In[66]:


print(reviewscv.toarray().shape)


# # 2. CountVectorizer with N-grams (Bigrams & Trigrams)

# In[67]:


cv_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
bow_matrix_ngram=cv_ngram_range.fit_transform(lemmas)


# In[68]:


print(cv_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# # 3. TF-IDF Vectorizer

# In[69]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv_ngram_max_features=TfidfVectorizer(norm='l2',analyzer='word',ngram_range=(1,3),max_features=500)
tfidf_matrix_ngram=tfidfv_ngram_max_features.fit_transform(lemmas)


# In[70]:


print(tfidfv_ngram_max_features.get_feature_names())
print(tfidf_matrix_ngram.toarray())


# # Generate Word Cloud

# In[71]:


# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(40,30))
    plt.imshow(wordcloud)
    plt.axis('off')

# Generate word cloud

STOPWORDS.add('Pron')
wordcloud=WordCloud(width=3000,height=2000,background_color='white',max_words=100,
                   colormap='Set2',stopwords=STOPWORDS).generate(clean_reviews)
plot_cloud(wordcloud)


# # Named Entity Recognition (NER)

# In[73]:


# Parts of speech (POS) tagging
nlp=spacy.load('en_core_web_sm')

one_block=clean_reviews
doc_block=nlp(one_block)
spacy.displacy.render(doc_block,style='ent',jupyter=True)


# In[75]:


for token in doc_block[50:100]:
    print(token,token.pos_)


# In[76]:


# Filtering the nouns and verbs only
nouns_verbs=[token.text for token in doc_block if token.pos_ in ('NOUN','VERB')]
print(nouns_verbs[50:100])


# In[77]:


# Counting the noun & verb tokens
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(nouns_verbs)
sum_words=X.sum(axis=0)

words_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
words_freq=sorted(words_freq,key=lambda x: x[1],reverse=True)

wd_df=pd.DataFrame(words_freq)
wd_df.columns=['word','count']
wd_df[0:10] # viewing top ten results


# In[78]:


# Visualizing results (Barchart for top 10 nouns + verbs)
wd_df[0:10].plot.bar(x='word',figsize=(12,8),title='Top 10 nouns and verbs');


# # Emotion Mining - Sentiment Analysis

# In[79]:


from nltk import tokenize
sentences=tokenize.sent_tokenize(' '.join(reviews))
sentences


# In[80]:


sent_df=pd.DataFrame(sentences,columns=['sentence'])
sent_df


# In[81]:


# Emotion Lexicon - Affin
affin=pd.read_csv('Afinn.csv',sep=',',encoding='Latin-1')
affin


# In[82]:


affinity_scores=affin.set_index('word')['value'].to_dict()
affinity_scores


# In[83]:


# Custom function: score each word in a sentence in lemmatised form, but calculate the score for the whole original sentence
nlp=spacy.load('en_core_web_sm')
sentiment_lexicon=affinity_scores

def calculate_sentiment(text:str=None):
    sent_score=0
    if text:
        sentence=nlp(text)
        for word in sentence:
            sent_score+=sentiment_lexicon.get(word.lemma_,0)
    return sent_score


# In[84]:


# manual testing
calculate_sentiment(text='good service')


# In[85]:


# Calculating sentiment value for each sentence
sent_df['sentiment_value']=sent_df['sentence'].apply(calculate_sentiment)
sent_df['sentiment_value']


# In[86]:


# how many words are there in a sentence?
sent_df['word_count']=sent_df['sentence'].str.split().apply(len)
sent_df['word_count']


# In[87]:


sent_df.sort_values(by='sentiment_value')


# In[88]:


# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


# In[89]:


# negative sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0]


# In[90]:


# positive sentiment score of the whole review
sent_df[sent_df['sentiment_value']>0]


# In[91]:


# Adding index cloumn
sent_df['index']=range(0,len(sent_df))
sent_df


# In[92]:


# Plotting the sentiment value for whole review
import seaborn as sns
plt.figure(figsize=(15,10))
sns.distplot(sent_df['sentiment_value'])


# In[93]:


# Plotting the line plot for sentiment value of whole review
plt.figure(figsize=(15,10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)

