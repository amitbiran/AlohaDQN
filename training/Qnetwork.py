import random
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM ,Input, Dense, Lambda
from keras.layers.merge import _Merge
from keras import backend as K
import numpy as np
import os
from keras.models import model_from_json

class QLayer(_Merge):
    '''Q Layer that merges an advantage and value layer'''
    def _merge_function(self, inputs):
        '''Assume that the inputs come in as [value, advantage]'''
        output = inputs[0] + (inputs[1] - K.mean(inputs[1], axis=1, keepdims=True))
        return output


class Qnetwork():
    def __init__(self,number_of_steps,features,number_of_lstm_units,OUTPUT_DIM):
        self.state = Input(shape=(number_of_steps, features))
        LSTM_layer = LSTM(units=number_of_lstm_units, return_sequences=False)(self.state)#make it with an abstract batch_size

       # Splice outputs of lastm layer using lambda layer
        x_value = Lambda(lambda LSTM_layer: LSTM_layer[ :, :number_of_lstm_units // 2])(LSTM_layer)
        x_advantage = Lambda(lambda LSTM_layer: LSTM_layer[:, number_of_lstm_units // 2:])(LSTM_layer)

       # Process spliced data stream into value and advantage function
        value = Dense(activation="linear", units=10)(x_value)
        advantage = Dense(activation="linear", units=10)(x_advantage)


        q = QLayer()([value, advantage])  # output dim is now 10
        self.q_out = Dense(activation="softmax", units=OUTPUT_DIM)(q)



        self.model = Model(inputs=[self.state], outputs=[self.q_out])
        self.model.compile(optimizer="Adam", loss="mean_squared_error")

        self.model.summary()

    def get_q(self, state):
        o=self.model.predict_on_batch(state)
        return o

    def sample(self, size):#need to keep last number of steps rewards and states for each one i sample
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

    def save_model(self,path):
        #save model hdf5
        self.model.save(os.path.join(path,"model.hdf5"))
        #save model as json
        model_json = self.model.to_json()
        with open(os.path.join(path,"model.json"), "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")

    def save_weights(self,path):
        self.model.save_weights(os.path.join(path, "weights.h5"))

    def load_weights(self,path):
        self.model.load_weights(os.path.join(path, "weights.h5"))

    def load_model(self,path):
        json_file = open(os.path.join(path,"model.json"), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        return loaded_model