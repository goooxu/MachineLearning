import numpy as np
from keras.models import Sequential
from keras.layers import Dense, RepeatVector, TimeDistributed, Activation, LSTM

DIGITS = 4
OPERATORS = 1
INPUT_LENGTH = DIGITS + (1 + DIGITS) * OPERATORS
OUTPUT_LENGTH = DIGITS + 1
CHARS = '0123456789+ '
char_indices = dict((c, i) for i, c in enumerate(CHARS))
indices_char = dict((i, c) for i, c in enumerate(CHARS))


def encode(strs, length):
    def encode_one(str, length):
        str += ' ' * (length - len(str))
        return np.eye(len(CHARS))[[char_indices[c] for c in str]]
    return np.array([encode_one(i, length) for i in strs])


def generate_dataset(size):
    array = np.random.randint(0, 10 ** DIGITS, size=(size, OPERATORS + 1))
    for nums in array:
        equation = '+'.join([str(i) for i in nums])
        result = str(eval(equation))
        yield (equation, result)


N = 50000
x_train, y_train = zip(*generate_dataset(N))
x_validation, y_validation = zip(*generate_dataset(N // 10))
x_train = encode(x_train, INPUT_LENGTH)
y_train = encode(y_train, OUTPUT_LENGTH)
x_validation = encode(x_validation, INPUT_LENGTH)
y_validation = encode(y_validation, OUTPUT_LENGTH)

HIDDEN_SIZE = 128


def build_model():
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE, input_shape=(
        INPUT_LENGTH, len(CHARS)), return_sequences=False))
    model.add(RepeatVector(OUTPUT_LENGTH))
    model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
    model.add(TimeDistributed(Dense(len(CHARS))))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


model = build_model()

MAX_EPOCHS = 100
BATCH_SIZE = 128
SAMPLES = 5

for iteration in range(0, MAX_EPOCHS):
    print('Iteration', iteration)
    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=1,
        validation_data=(x_validation, y_validation))

    for i in range(SAMPLES):
        x_present, y_present = zip(*generate_dataset(1))
        x_present = encode(x_present, INPUT_LENGTH)
        y_present = encode(y_present, OUTPUT_LENGTH)

        predictions = model.predict_classes(x_present, verbose=0)
        question = ''.join(indices_char[x]
                           for x in x_present[0].argmax(axis=-1))
        standard_answer = ''.join(indices_char[x]
                                  for x in y_present[0].argmax(axis=-1))
        predicted_answer = ''.join(indices_char[x] for x in predictions[0])
        print('Question         :', question)
        print('Standard  answer :', standard_answer)
        print('Predicted answer :', predicted_answer)
        print('---')
