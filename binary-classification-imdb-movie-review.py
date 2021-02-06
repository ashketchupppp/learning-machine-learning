# Binary Classification problem
# Classifying IMDB movie reviews as either positive or negative

from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

vocab_size = 10000
epochs = 4
batch_size = 512

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

def show_loss_and_accuracy(history, epochs):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    timeRange = range(1, epochs + 1)

    plt.figure(1)
    plt.plot(timeRange, loss_values, 'bo', label='Training Loss')
    plt.plot(timeRange, val_loss_values, 'b', label='Training Loss')
    plt.title('Training and validation loss')
    plt.xlabel('Timesteps')
    plt.ylabel('Loss')

    plt.figure(2)
    plt.plot(timeRange, acc_values, 'bo', label='Training Accuracy')
    plt.plot(timeRange, val_acc_values, 'b', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Timesteps')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()

# Obtain our testing and training data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# Vectorize and one-hot encode our training and test data
x_train = vectorize_sequences(train_data, vocab_size)
x_test = vectorize_sequences(test_data, vocab_size)
x_val = x_train[:vocab_size]
partial_x_train = x_train[vocab_size:]

# Vectorize our training and testing labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
y_val = y_train[:vocab_size]
partial_y_train = y_train[vocab_size:]

# Setup our model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(vocab_size,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Start training
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val))

# show_loss_and_accuracy(history, epochs)

# New strings to test on
strings = [
    'this was a rubbish movie I hated it',
    'this was the best movie I have ever seen'
]

word_index = imdb.get_word_index()
stringsAsIndexes = [string_to_word_index(x, word_index) for x in strings]
for i in range(0, len(stringsAsIndexes)):
    stringsAsIndexes[i] = [x for x in stringsAsIndexes[i] if x < vocab_size]
print(stringsAsIndexes)

vectorised_words = vectorize_sequences(stringsAsIndexes, vocab_size)
output = model.predict(vectorised_words)

for i in range(0, len(strings)):
    print(f"{strings[i]} : positivity rating is {output[i]}")
    