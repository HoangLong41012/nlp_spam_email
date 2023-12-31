import nltk
from collections import Counter

nltk.download('stopwords')
nltk.download('wordnet')

stopword = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

class PreProcess():
  def standardize(text):
    data = []
    for word in text.split():
        if word not in stopword:
            word = lemmatizer.lemmatize(word, "n")
            word = lemmatizer.lemmatize(word, "v")
            word = lemmatizer.lemmatize(word, "a")
            data.append(word)
    return ' '.join(data)

  def preProcess(text):
    line = text
    line = line.replace({'<.*?>': ''}, regex = True)
    line = line.replace({'[^A-Za-z]': ' '}, regex = True)
    line = line.str.replace('http\S+', ' ')
    line = line.str.replace('\W', ' ')
    line = line.str.replace('\d', ' ')
    line = line.str.lower()
    return line
  
  def createVocabulary(X_train):
    vocabulary = []
    for text in X_train:
      for word in text.split():
        vocabulary.append(word)

    vocabulary = []
    dictionary = Counter(vocabulary)
    for item in dictionary:
      if dictionary[item] >= 1:
        vocabulary.append(item)
    return vocabulary
  
  def filter(text, vocabulary):
    data = []
    for word in text.split():
        if word in vocabulary:
            data.append(word)
    return ' '.join(data)

  def transformLabel(label):
    if label == 'spam':
        return 1
    if label == 'ham':
        return 0