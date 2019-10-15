from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU, BatchNormalization
from keras.models import Model
from functions import *
from matplotlib.animation import FuncAnimation
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import keras 

dataset_path="..\\data_2033\\ecg_data_200.json" #файл с датасетом
data = Openf(dataset_path) # Open data file
print("file was opened")

count_of_diffrent_signals = 10
count_augmentation_data = 10
batch_size = count_of_diffrent_signals*count_augmentation_data
count_of_batch = 15
size_of_data = 3000
array_of_pooling = [2,5,2]
SIZE = 5000

my_generator = Create_data_generator(data = data,
									count_augmentation_data =count_augmentation_data,
									count_of_diffrent_signals=count_of_diffrent_signals,
									size_of_data= size_of_data, flag = "train") # создаём генератор

my_generator_test = Create_data_generator(data = data,
									count_augmentation_data =count_augmentation_data,
									count_of_diffrent_signals=count_of_diffrent_signals,
									size_of_data= size_of_data, flag = "test") # создаём генератор


#########################################
## CREATE ENCODER and DECODER CEPARELLY
#########################################

x_plac = Input(batch_shape=(None,size_of_data,1))  # adapt this if using `channels_first` image data format

x = Conv1D(15, 100, activation=LeakyReLU(alpha = 0.2), padding='same')(x_plac)
x = MaxPooling1D(array_of_pooling[0], padding='same')(x)
x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
x = Conv1D(10, 80,activation=LeakyReLU(alpha = 0.2), padding='same')(x)
x = MaxPooling1D(array_of_pooling[1], padding='same')(x)
x = BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001) (x)
x = Conv1D(5, 30, activation=LeakyReLU(alpha = 0.2), padding='same')(x)
encoded = MaxPooling1D(array_of_pooling[2], padding='same')(x)

encoding_dim = SIZE//array_of_pooling[0]//array_of_pooling[1]//array_of_pooling[2]

encoder = Model(x_plac, encoded) 

encoded_input = Input(batch_shape=(None,encoding_dim,5))
x = UpSampling1D(array_of_pooling[2])(encoded_input)
x = Conv1D(5, 30, activation=LeakyReLU(alpha = 0.2), padding='same')(x)
x = UpSampling1D(array_of_pooling[1])(x)
x = Conv1D(10, 80, activation=LeakyReLU(alpha = 0.2), padding='same')(x)
x = UpSampling1D(array_of_pooling[0])(x)
decoded = Conv1D(1, 100, activation=LeakyReLU(alpha = 0.2), padding='same')(x)

decoder = Model(encoded_input, decoded)

auto_input = Input(batch_shape=(None,size_of_data,1))
encoded = encoder(auto_input)
decoded = decoder(encoded)
auto_encoder = Model(auto_input, decoded)


tensorboard = TensorBoard(log_dir = './graphs1')

#########################################
## LEARNING
#########################################
auto_encoder.compile(optimizer=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95), loss='mean_squared_error')

np.shape(next(my_generator))

history = auto_encoder.fit_generator(my_generator,validation_data = my_generator_test, validation_steps =10,
                    steps_per_epoch=count_of_batch, epochs=20)

visualize_learning(history, 'loss','val_loss')


###############################
## VISUALIZE на уровне - ТОПОР
##############################
plt.figure(1)
data1 = next(my_generator) # take a batch
plt.subplot(2,1,1)
data_ = np.reshape(data1[0][0],(size_of_data))
Print_EKG(data_)
output = auto_encoder.predict(data1[0])
plt.subplot(2,1,2)
data2 = np.reshape(output[0],(size_of_data))
Print_EKG(data2)
plt.show()


########################################
## Генерирование данных и запись в файл
########################################
# Generate_bad_data(500,"data",encoder, decoder, data,size_of_data)




