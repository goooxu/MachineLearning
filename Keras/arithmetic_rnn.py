import numpy as np
from math import factorial
from itertools import permutations
from keras.models import Model
from keras.layers import Input, Dense, RepeatVector, TimeDistributed, Activation, LSTM, Concatenate, Multiply, Flatten, Reshape, Permute

DIGITS = 3
OPERATORS = 1
INPUT_LENGTH = DIGITS + (1 + DIGITS) * OPERATORS
OUTPUT_LENGTH = DIGITS * (OPERATORS + 1)
CHARS = '0123456789* '
char_indices = dict((c, i) for i, c in enumerate(CHARS))
indices_char = dict((i, c) for i, c in enumerate(CHARS))


def encode(strs, length):
    def encode_one(str, length):
        str += ' ' * (length - len(str))
        return np.eye(len(CHARS), dtype=np.float32)[[char_indices[c] for c in str]]
    return np.array([encode_one(i, length) for i in strs])


def decode(array):
    return ''.join(indices_char[x] for x in array.argmax(axis=-1))


def generate_dataset(size):
    size = size // factorial(OPERATORS + 1)
    array = np.random.randint(0, 10 ** DIGITS, size=(size, OPERATORS + 1))
    for nums in array:
        result = None
        for permutation in permutations(nums):
            equation = '*'.join([str(i) for i in permutation])
            if not result:
                result = str(eval(equation))
            yield (equation, result)


TRAIN_SET_SIZE = 500000
VALIDATION_SET_SIZE = 5000
x_train, y_train = zip(*generate_dataset(TRAIN_SET_SIZE))
x_validation, y_validation = zip(*generate_dataset(VALIDATION_SET_SIZE))
x_train = encode(x_train, INPUT_LENGTH)
y_train = encode(y_train, OUTPUT_LENGTH)
x_validation = encode(x_validation, INPUT_LENGTH)
y_validation = encode(y_validation, OUTPUT_LENGTH)

HIDDEN_SIZE = 1024


def attention(inputs, length):
    layer = Permute((2, 1))(inputs)
    layer = Dense(length, activation='softmax')(layer)
    layer = Permute((2, 1))(layer)
    return Multiply()([inputs, layer])


def build_model():
    inputs = Input(shape=(INPUT_LENGTH, len(CHARS)))
    layer = LSTM(HIDDEN_SIZE, return_sequences=False)(inputs)
    layer = RepeatVector(OUTPUT_LENGTH)(layer)
    layer = LSTM(HIDDEN_SIZE, return_sequences=True)(layer)
    layer = attention(layer, OUTPUT_LENGTH)
    layer = TimeDistributed(Dense(len(CHARS)))(layer)
    outputs = Activation('softmax')(layer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


model = build_model()

MAX_EPOCHS = 100
BATCH_SIZE = 100
SAMPLES = 1

max_val_acc = .0

for iteration in range(0, MAX_EPOCHS):
    print('Iteration', iteration)
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=1,
        validation_data=(x_validation, y_validation))

    val_acc = history.history['val_acc'][0]
    max_val_acc = max(max_val_acc, val_acc)
    print('Max val acc : {:0.4f}'.format(max_val_acc))
    if val_acc < max_val_acc * 0.90:
        break

    x_present, y_present = zip(
        *generate_dataset(SAMPLES * factorial(OPERATORS + 1)))
    x_present = encode(x_present, INPUT_LENGTH)
    y_present = encode(y_present, OUTPUT_LENGTH)
    predictions = model.predict(x_present, verbose=0)

    for i in range(SAMPLES * factorial(OPERATORS + 1)):
        print('Question         :', decode(x_present[i]))
        print('Standard  answer :', decode(y_present[i]))
        print('Predicted answer :', decode(predictions[i]))
        print('---')
