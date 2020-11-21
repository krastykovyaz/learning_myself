import nltk

text = "I want to apply a credit's mortgage?"

token = nltk.tokenize.WhitespaceTokenizer()
print(token.tokenize(text))

token = nltk.tokenize.TreebankWordTokenizer()
print(token.tokenize(text))

token = nltk.tokenize.WordPunctTokenizer()
print(token.tokenize(text))

text = "feet wolves cats talked"
tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)
stemmer = nltk.stem.PorterStemmer()
print(" ".join(stemmer.stem(token) for token in tokens))
stemmer = nltk.stem.WordNetLemmatizer()
print(" ".join(stemmer.lemmatize(token) for token in tokens))



