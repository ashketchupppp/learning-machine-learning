from keras.datasets import imdb
import numpy as np

vocab_size = 10000

# One-hot encode sequences of words
def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results

def string_to_word_index(string, word_index):
    stringSplitBySpaces = string.split(' ')
    wordList = []
    for word in stringSplitBySpaces:
        try:
            wordList.append(word_index[word])
        except:
            pass
    return wordList

# New strings to test on
strings = [
    'this was a rubbish movie I hated it',
    'this was the best movie I have ever seen'
]

word_index = imdb.get_word_index()
stringsAsIndexes = [string_to_word_index(x, word_index) for x in strings]
for string in stringsAsIndexes:
    string = [x for x in string if x < vocab_size]
print(stringsAsIndexes)

vectorised_words = vectorize_sequences(stringsAsIndexes, vocab_size)