import numpy as np 
import nltk 
from nltk.stem.porter import PorterStemmer 

nltk.download('punkt')
stemmer = PorterStemmer() 

kalimat = 'HallO, HoW are you ?'
word = 'organizing'

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bow(tokenize, words):
    sentence = [stem(word) for word in tokenize]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence:
            bag[idx] = 1
    return bag


    




