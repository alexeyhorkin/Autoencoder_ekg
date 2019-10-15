import json
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import random


dataset_path="data_2033\\data_2033.json" #файл с датасетом
def Openf(link):
    # function for opening some json files
    with open (link,'r') as f:
        data = json.load(f)
        return data

def Create_data_generator(data,count_augmentation_data,count_of_diffrent_signals,size_of_data,flag):
    # generator for learning data with augmentation
    rd.seed(6)
    length_of_move = 10 # step of moving
    SIZE = 5000
    procent_train = 0.8
    count_of_train = int(len(data.keys())*procent_train)
    data_train = { k: data[k] for k in list(data.keys())[:count_of_train]} 
    data_test =  { k: data[k] for k in list(data.keys())[count_of_train:]}
    while True:
        RES = []
        count = 0
        if flag == "train":
            DATA = data_train
        elif flag == "test":
            DATA = data_test
        for i in range(count_of_diffrent_signals):
            leads = data_train[str(rd.sample(data_train.keys(), 1)[0])]["Leads"] # take random a patient
            otvedenie = str(rd.sample(leads.keys(),1)[0])
            signal = leads[otvedenie]["Signal"] # take a signal 
            start = rd.randint(0,SIZE - size_of_data-count_augmentation_data*length_of_move)
            for x in range(count_augmentation_data+1): #делаем срезы по каждому пациенту
                res = signal[start+x*length_of_move : start+x*length_of_move + size_of_data] # make a slice
                RES.append(res) # add resault in batch
                count +=1
        RES = np.array(RES)
        RES = np.reshape(RES, (count,size_of_data,1))
        yield (RES,RES)

def Create_data_for_classificator(data_good_ekg, data_false_ekg,size_of_batch,size_of_data,flag):
    #generator for classificator bad and good ekg
    # 1 - bad ecg
    # 0 - good ecg
    rd.seed(6)
    length_of_move = 10 # step of moving
    procent_train = 0.8
    count_of_train_false = int(len(data_false_ekg.keys())*procent_train)
    count_of_train_true = int(len(data_good_ekg.keys())*procent_train)
    print("__________ false - ",count_of_train_false)
    print("__________ true - ",count_of_train_true)
    data_train_false_ekg = { k: data_false_ekg[k] for k in list(data_false_ekg.keys())[:count_of_train_false]} 
    data_test_false_ekg =  { k: data_false_ekg[k] for k in list(data_false_ekg.keys())[count_of_train_false:]}
    data_train_good_ekg = { k: data_good_ekg[k] for k in list(data_good_ekg.keys())[:count_of_train_true]} 
    data_test_good_ekg =  { k: data_good_ekg[k] for k in list(data_good_ekg.keys())[count_of_train_true:]}

    SIZE1 = len(data_train_false_ekg[0])
    SIZE2 = 5000 
    if flag == "train":
        DATA_bad = data_train_false_ekg
        DATA_good = data_train_good_ekg 
    elif flag == "test":
        DATA_bad = data_test_false_ekg
        DATA_good = data_test_good_ekg
    while True:
        RES = []
        labels = []
        count = 0
        for i in range(size_of_batch):
            start1, start2   =   ( rd.randint(0,SIZE1 - size_of_data) , rd.randint(0,SIZE2 - size_of_data) ) 
            index1 , index2  =   ( rd.sample(DATA_bad.keys(),1)[0] , rd.sample(DATA_good.keys(), 1)[0]  ) 
            otvedenie        =   rd.sample(    DATA_good[index2]["Leads"].keys() ,1)[0]
            signal1, signal2 =   ( DATA_bad[index1], DATA_good[index2]["Leads"][otvedenie]["Signal"] )
            res1 , res2      =   ( signal1[start1 : start1 + size_of_data] , signal2[start2 : start2 + size_of_data] ) 
            RES.append(res1) # add a label 1
            RES.append(res2) # add a label 0
            labels.append([1])
            labels.append([0])
            count+=2 
        RES = np.array(RES)
        RES = np.reshape(RES, (count,size_of_data,1))
        labels = np.array(labels)
        yield (RES,labels)

def Print_EKG(signal):
    size = len(signal)
    plt.plot(range(size),signal)
    plt.xlabel(r'$time$' ,  fontsize=15, horizontalalignment='right' , x=1)
    plt.ylabel(r'$value$',  fontsize=15, horizontalalignment='right',  y=1)

def visualize_latent_space(start,end,decoder,count_of_step,size_of_data):
    from matplotlib.animation import FuncAnimation
    direction = (end-start)/count_of_step
    RES = []
    for i in range(count_of_step+1):
        tmp = start + i*direction
        RES.append(tmp)
    RES = np.array(RES) # this is our batch
    print(np.shape(RES))

    final_out = decoder.predict(RES)

    fig = plt.figure(3)
    ax1 = fig.add_subplot(1, 1, 1)
    def animate(i):
        x = np.arange(0,size_of_data)
        y = final_out[i].reshape(size_of_data)
        ax1.clear()
        ax1.plot(x, y)
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title("iteration "+ str(i)+ "/"+ str(count_of_step+1))
    anim = FuncAnimation(fig, animate,frames=count_of_step+1, interval=30)
    anim.save('animation_1.gif', writer='imagemagick', fps=60)
    plt.show()


def Get_bad_EKG(ecg1,ecg2,ecg3,decoder):
    alpha1 = random.normalvariate(0.5,0.2)
    alpha2 = random.normalvariate(0.5,0.2)
    alpha3 = random.normalvariate(0.5,0.2)
    data1 = alpha1*ecg1 + (1-alpha1)*ecg2
    data2 = alpha2*ecg1 + (1-alpha2)*ecg3
    data3 = alpha3*ecg2 + (1-alpha3)*ecg3
    tmp = np.array([data1,data2,data3])
    return decoder.predict(tmp)

def Generate_bad_data(count, path, encoder , decoder, data, size_of_data):
    RES = {}
    generator = Create_data_generator(data = data,
                                        count_augmentation_data =0,
                                        count_of_diffrent_signals=3,
                                        size_of_data= size_of_data, 
                                        flag = "train") # создаём генератор

    index = 0
    for i in range(count):
        data_for_denerator = next(generator)[0]
        ecg1, ecg2, ecg3 = encoder.predict(data_for_denerator)
        bad = Get_bad_EKG(ecg1, ecg2, ecg3, decoder)
        bad =  np.reshape(bad, (3,size_of_data))
        for j in range(3):
            RES[index] = bad[j]
            index = index+1
            print(str(index) + "/"+ str(count*3) + " saved" )

    np.save(path+'.npy', RES)        



def Load_bad_EKG(path):
    return np.load(path).item()

def visualize_learning(history, graph1, graph2 ):
    plt.plot(history.history[graph1])
    plt.plot(history.history[graph2])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid()
    plt.show()