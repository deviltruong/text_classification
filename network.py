from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Masking

def building_network(num_lstm_layer, num_hidden_node, dropout, time_step, vector_length, output_length):
    model = Sequential()
    model.add(LSTM(num_hidden_node, return_sequences=True, dropout=dropout,
                   input_shape=(time_step, vector_length)))
    for i in range(num_lstm_layer - 1):
        model.add(LSTM(num_hidden_node, return_sequences=True, dropout=dropout))

    model.add(Flatten())

    model.add(Dense(output_length, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model