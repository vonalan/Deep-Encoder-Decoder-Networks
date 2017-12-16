from keras.models import Model
from keras.layers import Input, LSTM, ConvLSTM2D, LSTMCell, StackedRNNCells

# TODO: StackedRNNCells
input_stream_shape = (15, 96, 96, 3)
inputs = Input(input_stream_shape)
x = ConvLSTM2D(3, kernel_size=(1,1), return_sequences=True, go_backwards=False,)(inputs)
x = ConvLSTM2D(3, kernel_size=(1,1), return_sequences=True, go_backwards=False)(x)
outputs = ConvLSTM2D(3, kernel_size=(1,1), return_sequences=True, go_backwards=False)(x)
model = Model(inputs, outputs)
for i, layer in enumerate(model.layers):
    print(i, layer.output_shape)

# # TODO: StackedRNNCells
# input_stream_shape = (15, 1024)
# inputs = Input(input_stream_shape)
# x = LSTM(1024, return_sequences=True, go_backwards=True)(inputs)
# x = LSTM(1024, return_sequences=True, go_backwards=True)(x)
# outputs = LSTM(1024, return_sequences=True, go_backwards=True)(x)
# model = Model(inputs, outputs)
# for i, layer in enumerate(model.layers):
#     print(i, layer.output_shape)

