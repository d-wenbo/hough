import numpy as np
import cv2
import random
import os
import sys
import glob
import itertools
import matplotlib.pyplot as plt
import math
import csv
args = sys.argv
datafile = 'csv/a.csv' 

new_dir_path_graph = 'hough_hist/'
os.makedirs(new_dir_path_graph,exist_ok = True)


def rms(list):
    np_list = np.array(list)
    np_list = np.square(np_list)
    mse = np.sum(np_list)/(np_list.size-1)
    return mse

if __name__ == "__main__":
    
    '''
    files = glob.glob(datafile + "/*.csv")
    files.sort()
    f1 = open('result_non9.csv', 'w')
    writer = csv.writer(f1, lineterminator='\n')
    writer.writerow(['para','rms'])
    '''
    ans_xc = []
    ans_yc = []
    ans_hex = []
    ans_hey = []


    with open('ans_1.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            ans_xc.append(row[0])
            ans_yc.append(row[1])
            ans_hex.append(row[2])
            ans_hey.append(row[3])
    f.close

    del ans_xc[0]
    del ans_yc[0]
    del ans_hex[0]
    del ans_hey[0]
    
    
    

    ans_xc_f = [float(i) for i in ans_xc ]
    ans_yc_f = [float(i) for i in ans_yc ]
    ans_hex_f = [float(i) for i in ans_hex ]
    ans_hey_f = [float(i) for i in ans_hey]


    file_1 = datafile
    



    

    data_xc = []
    data_yc = []
    data_hex = []
    data_hey = []



    
    with open(file_1) as d:
        reader = csv.reader(d)
        for row in reader:
            data_xc.append(row[0])
            data_yc.append(row[1])
            data_hex.append(row[2])
            data_hey.append(row[3])
    d.close

    
    

    del data_xc[0]
    del data_yc[0]
    del data_hex[0]
    del data_hey[0]
    
    
    
    

    data_xc_f = [float(i) for i in data_xc ]
    data_yc_f = [float(i) for i in data_yc ]
    data_hex_f = [float(i) for i in data_hex ]
    data_hey_f = [float(i) for i in data_hey]


    list_dx_cs = []
    list_dy_cs = []
    list_dx_he = []
    list_dy_he = []
    for a_xc,a_yc,a_hex,a_hey,d_xc,d_yc,d_hex,d_hey in zip(ans_xc_f,ans_yc_f,ans_hex_f,ans_hey_f,data_xc_f,data_yc_f,data_hex_f,data_hey_f):
        
        list_dx_cs.append(d_xc-a_xc)
        list_dy_cs.append(d_yc-a_yc)
        list_dx_he.append(d_hex-a_hex)
        list_dy_he.append(d_hey-a_hey)
    a = rms(list_dx_cs)
    b = rms(list_dy_cs)
    c = rms(list_dx_he)
    d = rms(list_dy_he)
    #print(a,b,c,d)    


        
     
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(2,2,1)
    ax1.hist(list_dx_cs,range=(-20,20),bins = 40)
    ax1.set_title("dx_cs",fontsize=12)
    plt.ylim(0,10)
    
    ax2 = fig.add_subplot(2,2,2)
    ax2.hist(list_dy_cs,range=(-20,20),bins = 40)
    ax2.set_title("dy_cs",fontsize=12)
    plt.ylim(0,10)


    ax3 = fig.add_subplot(2,2,3)
    ax3.hist(list_dx_he,range=(-20,20),bins = 20)
    ax3.set_title("dx_he",fontsize=12)
    plt.ylim(0,10)
    ax4 = fig.add_subplot(2,2,4)
    ax4.hist(list_dy_he,range=(-20,20),bins = 20)
    ax4.set_title("dy_he",fontsize=12)
    plt.ylim(0,10) 
    plt.tight_layout()
    plt.show()


    fig.savefig(new_dir_path_graph + 'hist.png')
    
                