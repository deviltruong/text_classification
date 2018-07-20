from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Embedding


def building_network(vocab_size, embedding_size, num_lstm_layer, num_hidden_node,
                     dropout, time_step, output_lenght):
    model = Sequential()

    model.add(Embedding(vocab_size, embedding_size, input_length=time_step, mask_zero=False))

    for i in xrange(num_lstm_layer):
        model.add(LSTM(num_hidden_node, return_sequences=True, dropout=dropout))

    model.add(Flatten())

    model.add(Dense(output_lenght, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model