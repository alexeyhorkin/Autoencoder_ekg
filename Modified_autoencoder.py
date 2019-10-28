from functions import *
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU, BatchNormalization,Flatten, Dropout, Lambda, Add, add
from keras.models import Model
from matplotlib.animation import FuncAnimation
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import keras
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
path = "data.npy"


data_bad = Load_bad_EKG(path)

dataset_path="..\\data_2033\\ecg_data_200.json" #файл с датасетом
data = Openf(dataset_path) # Open data file
print("file was opened")
size_of_data = 2000
count_of_batch = 30
size_of_batch = 10
drop_out_rate = 0.3
generator = Create_data_for_classificator(data,data_bad,size_of_batch,size_of_data,"train")
generator_test = Create_data_for_classificator(data,data_bad,size_of_batch,size_of_data,"test")

def create_classificator(input_for_classificator):
	x = Conv1D(15, 100, activation=LeakyReLU(alpha = 0.2), padding='same')(input_for_classificator)
	x = MaxPooling1D(2, padding='same')(x)
	x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
	x = Conv1D(10, 80,activation=LeakyReLU(alpha = 0.2), padding='same')(x)
	x = MaxPooling1D(5, padding='same')(x)
	x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
	x = Conv1D(20, 30, activation=LeakyReLU(alpha = 0.2), padding='same')(x)
	x = MaxPooling1D(2, padding='same')(x)
	x = Flatten()(x)
	x = Dense(30, activation='sigmoid')(x)
	x = Dropout(drop_out_rate)(x)
	x = Dense(15, activation='sigmoid')(x)
	x = Dropout(drop_out_rate)(x)
	out = Dense(1, activation='sigmoid')(x) 
	return Model(input_for_classificator, out) 

def create_encoder(array_of_pooling, input_for_encoder):
	'''create an encoder'''
	x = Conv1D(15, 100, activation=LeakyReLU(alpha = 0.2), padding='same')(input_for_encoder)
	x = MaxPooling1D(array_of_pooling[0], padding='same')(x)
	x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
	x = Conv1D(10, 80,activation=LeakyReLU(alpha = 0.2), padding='same')(x)
	x = MaxPooling1D(array_of_pooling[1], padding='same')(x)
	x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
	x = Conv1D(5, 30, activation=LeakyReLU(alpha = 0.2), padding='same')(x)
	encoded = MaxPooling1D(array_of_pooling[2], padding='same')(x)
	return Model(input_for_encoder, encoded) # create a model 

def create_decoder(array_of_upsampling, input_for_decoder):
	'''create a decoder'''
	x = UpSampling1D(array_of_upsampling[0])(input_for_decoder)
	x = Conv1D(5, 30, activation=LeakyReLU(alpha = 0.2), padding='same')(x)
	x = UpSampling1D(array_of_upsampling[1])(x)
	x = Conv1D(10, 80, activation=LeakyReLU(alpha = 0.2), padding='same')(x)
	x = UpSampling1D(array_of_upsampling[2])(x)
	decoded = Conv1D(1, 100, activation=LeakyReLU(alpha = 0.2), padding='same')(x)
	return Model(input_for_decoder, decoded) # create a model

def create_autoencoder(input_for_autoencoder, input_for_encoder,
						input_for_decoder,array_of_pooling, 
						array_of_upsampling):
	'''create encoder + decoder''' 
	encoder = create_encoder(array_of_pooling, input_for_encoder)
	decoder = create_decoder(array_of_upsampling, input_for_decoder)
	decoded = decoder(encoder(input_for_autoencoder))
	return Model(input_for_autoencoder, decoded)

def create_big_model( input_for_model, encoder, decoder, classificator):
	x = encoder(input_for_model)
	x1 = Lambda( lambda x: K.slice(x, (0, 0, 0), (1, -1, -1)))(x)
	x2 = Lambda( lambda x: K.slice(x, (1, 0, 0), (1, -1, -1)))(x)
	center = Lambda (lambda x: x*0.5)( Add()([x1, x2]) ) 
	psevdo_ecg = decoder(center)
	out_of_classificator = classificator(psevdo_ecg)
	# print("out_of_classificator")
	# print(out_of_classificator)
	out_of_decoder = decoder(x)
	return  Model(inputs=input_for_model,outputs=[out_of_decoder, out_of_classificator])


###############################################
## Building a model
###############################################
SIZE = 5000
size_of_data = 2000
array_of_pooling = [2,5,2]
array_of_upsampling = array_of_pooling
encoding_dim = SIZE//array_of_pooling[0]//array_of_pooling[1]//array_of_pooling[2]
count_of_batch = 15

# define inputs placeholders for all models
input_for_autoencoder = Input(batch_shape=(2,size_of_data,1) )
input_for_encoder = Input(batch_shape=(2,size_of_data,1) )
input_for_decoder = Input(batch_shape=(2,encoding_dim,5) )
input_for_classificator = Input(batch_shape = (None,size_of_data,1))

# work with classificator
classificator = create_classificator(input_for_classificator)
generator_for_cl = Create_data_for_classificator(data,data_bad,size_of_batch,size_of_data,"train")
classificator.compile(optimizer=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95), loss='binary_crossentropy',metrics=['accuracy'])
history_of_classificator = classificator.fit_generator(generator_for_cl, validation_data = generator_test, validation_steps =10,
                    steps_per_epoch=count_of_batch, epochs=20)

visualize_learning(history_of_classificator, 'accuracy','val_accuracy')

# forbit training for weights
for l in classificator.layers:
    l.trainable = False

# print some weights
l = classificator.get_layer(index  = 3)
w = l.get_weights()
print("weights")
print(w)

# work with the whole model
encoder = create_encoder(array_of_pooling, input_for_encoder)
decoder = create_decoder(array_of_upsampling, input_for_decoder) 

my_generator = Create_data_generator_for_model(data = data,
				size_of_data= size_of_data, flag = "train") # создаём генератор
BIG_MODEL = create_big_model(input_for_autoencoder , encoder,decoder, classificator)
my_loss = ["mse" , "mse"]
lossWeights = [1.0, 1.0]
BIG_MODEL.compile(optimizer=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95), loss=my_loss, loss_weights=lossWeights)

history = BIG_MODEL.fit_generator(my_generator,
                    steps_per_epoch=count_of_batch, epochs=20)

visualize_learning_model(history, 'model_1_loss', 'model_3_loss')

# l = classificator.get_layer(index  = 3)
w = l.get_weights()
print("weights")
print(w)

