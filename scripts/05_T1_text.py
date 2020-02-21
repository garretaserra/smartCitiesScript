import pandas as pd

#
# All in one class
#

texts = [
     'This is the first document.',
     'This is the second second document.',
     'And the third one.',
     'Is this the first document?',
]

# Counting
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, stop_words='english')
X = vectorizer.fit_transform(texts)
print(X.shape)
print(vectorizer.get_feature_names())
results = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
print(results)


# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
print(X.shape)
print(vectorizer.get_feature_names())
results = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
print(results)

# Hashing for memory mapping
from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = HashingVectorizer(stop_words='english')
X = vectorizer.transform(texts)
print(X.shape)
print(vectorizer.get_feature_names())
results = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
print(results)


# Using external tools
import nltk
nltk.download('punkt')
import nltk.corpus
nltk.download('stopwords')
stop = nltk.corpus.stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
    
vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), strip_accents='ascii', lowercase=True, stop_words=stop)  
X = vectorizer.fit_transform(texts)
print(X.shape)
print(vectorizer.get_feature_names())
results = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
print(results)

#
# Data transformation step by step
#

# conda install -c anaconda nltk
import nltk
nltk.download('punkt')
import nltk.corpus


text = 'This I is the! first document waiting.'

#Lower case
text = "".join(x.lower() for x in text.split('. '))

# Removing Punctuation
import string
import re
text = re.sub('['+string.punctuation+']', '', text)

# Word tokenize
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)

# stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
tokens = [x for x in tokens if x not in stop]

# Stemmer
from nltk.stem import PorterStemmer
pst = PorterStemmer()
tokens = [pst.stem(x) for x in tokens]

#  Lemmatizer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer() 
tokens = [lemmatizer.lemmatize(x) for x in tokens]

# Counting
from nltk.probability import FreqDist
counting = FreqDist(tokens)



