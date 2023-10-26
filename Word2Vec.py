# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import nltk
nltk.download('punkt')

from gensim.models import Word2Vec
from nltk.corpus import stopwords

nltk.download('stopwords')

import re

paragraph ="""The sky was full of stars, and the moon was shining bright. I was walking on the terrace of my house, and I could see the lights of Rameswaram town in the distance. I was thinking about my life, my dreams, and my future. I had always been fascinated by the sky and the stars, and I had dreamed of flying since I was a child. I wanted to be a pilot, but my family could not afford to send me to flying school. Instead, I studied engineering and became a scientist. But I never forgot my dream of flying. When I was working on India’s missile program, I realized that I could make my dream come true. I could design and build a plane that would fly higher than any other plane in the world. And so I did. The plane was called the Nandi, and it flew higher than any other plane in the world. It was a great achievement, but it was only the beginning. I went on to work on India’s space program, and we launched our first satellite into space in 1975. It was an incredible feeling to know that we had done something that no one else in India had ever done before. But again, it was only the beginning. We went on to launch more satellites, and we sent our first astronaut into space in 1984. It was an incredible journey, and it all started with a dream."""

#preprocessing of the data

text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

#preparing the dataset
sentences = nltk.sent_tokenize(text)
sentences = [ nltk.word_tokenize(sentence) for sentence in sentences]
for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
  
#Training the Word2Vec Model

model = Word2Vec(sentences, min_count=1)
words = model.wv.key_to_index

#Finding Word Vectors




