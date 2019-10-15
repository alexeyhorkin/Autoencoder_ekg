from functions import *
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU, BatchNormalization,Flatten, Dropout
from keras.models import Model
from matplotlib.animation import FuncAnimation
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import keras  
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
tensorboard = TensorBoard(log_dir = './graphs2')

#########################################
## CREATE MODEL
#########################################


x_plac = Input(batch_shape=(None,size_of_data,1))  # adapt this if using `channels_first` image data format

x = Conv1D(15, 100, activation=LeakyReLU(alpha = 0.2), padding='same')(x_plac)
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
encoded = Dense(1, activation='sigmoid')(x) 

classificator = Model(x_plac, encoded) 

classificator.compile(optimizer=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95), loss='binary_crossentropy',metrics=['accuracy'], )
history = classificator.fit_generator(generator, validation_data = generator_test, validation_steps =10,
                    steps_per_epoch=count_of_batch, epochs=20)

visualize_learning(history, 'accuracy','val_accuracy')



