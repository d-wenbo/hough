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
import pickle


args = sys.argv
filename = 'mask_only_score1'
f = open('hough.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(['c(x)','cs(y)', 'he(x)','he(y)'])

def distance (x1,y1,x2,y2):
    distance = math.sqrt(((x1-x2)**2)+((y1-y2)**2))
    return distance
def hough_transform(img):
    rho = 1
    theta = np.pi/180
    hough_thre = 80
    minLineLength = 30
    maxLineGap = 50
    dst_line= cv2.HoughLinesP(img,
    rho,
    theta,
    hough_thre,
    minLineLength,
    maxLineGap)
    return dst_line
def angle(x1,y1,x2,y2):
    if x2>x1:
        phi = math.atan2((y2 - y1),(x2 - x1 + 1e-8)) 
        
    else:
        phi = math.atan2((y1 - y2),(x1 - x2 + 1e-8)) 
    return phi

def linesolve_inv(x1,y1,x2,y2):
    left = np.array([[x1,1],[x2,1]])
    right = np.array([y1,y2])
    left_inv = np.linalg.pinv(left)
    solve = np.dot(left_inv,right)
    a = solve[0]
    b = solve[1]
    return a,b

def cs_solve(a1,b1,a2,b2):
    left = np.array([[a1,-1],[a2,-1]])
    right = np.array([-b1,-b2])
    [x_c,y_c] = np.linalg.solve(left, right)
    return x_c,y_c

def list_ave(list):
    ave = sum(list)/len(list)
    return ave

def unitvec(x1_1,y1_1,x2_1,y2_1,
x1_2,y1_2,x2_2,y2_2,
x_c,y_c):
    l_max1 = distance(x1_1,y1_1,x2_1,y2_1)
    l_max2 = distance(x1_2,y1_2,x2_2,y2_2)
    l1_1 = distance(x1_1,y1_1,x_c,y_c)
    l2_1 = distance(x2_1,y2_1,x_c,y_c)
    l1_2 = distance(x1_2,y1_2,x_c,y_c)
    l2_2 = distance(x2_2,y2_2,x_c,y_c)
    if l1_1 > l2_1:
        unitvec_x1 = (x1_1-x2_1)/l_max1
        unitvec_y1 = (y1_1-y2_1)/l_max1
    else:
        unitvec_x1 = (x2_1-x1_1)/l_max1
        unitvec_y1 = (y2_1-y1_1)/l_max1
    if l1_2 > l2_2:
        unitvec_x2 = (x1_2-x2_2)/l_max2
        unitvec_y2 = (y1_2-y2_2)/l_max2
    else:
        unitvec_x2 = (x2_2-x1_2)/l_max2
        unitvec_y2 = (y2_2-y1_2)/l_max2
    
    return unitvec_x1,unitvec_y1,unitvec_x2,unitvec_y2

if __name__ == "__main__":


    imgs = glob.glob(filename + "/*.png")
    imgs.sort()
    #print(imgs)
    new_dir_path_hough = 'hough_only_2/'
    os.makedirs(new_dir_path_hough,exist_ok = True)
    new_dir_path_line_sep = 'hough_only_line_sep_2/'
    os.makedirs(new_dir_path_line_sep,exist_ok = True)
    new_dir_path_line_sep_1line = 'hough_only_line_sep_1line_2/'
    os.makedirs(new_dir_path_line_sep_1line,exist_ok = True)
    new_dir_path_graph_judge = 'graph_judge_2/'
    os.makedirs(new_dir_path_graph_judge,exist_ok = True)

    list_coordinate = []

    i = 0
    while (i < 37):
        img = cv2.imread(imgs[i], 0)
        img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_3ch_1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_3ch_2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, img_thre = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
        
        coordinate = {}
        coordinate["id"] = i
        
        dst_line = hough_transform(img_thre)
        #print(dst_line)
        list_x1 = []
        list_y1 = []
        list_x2 = []
        list_y2 = [] 
        line_1_x1 = []
        line_1_y1 = []
        line_1_x2 = []
        line_1_y2 = []
        line_2_x1 = []
        line_2_y1 = []
        line_2_x2 = []
        line_2_y2 = []
        
        for line in dst_line:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_3ch,(x1,y1),(x2,y2),(0,0,255),1)
            list_x1.append(x1)
            list_y1.append(y1)
            list_x2.append(x2)
            list_y2.append(y2)
        x_1 = list_x1[0]
        y_1 = list_y1[0]
        x_2 = list_x2[0]
        y_2 = list_y2[0]
        phi = angle(x_1,y_1,x_2,y_2)

        for x1,y1,x2,y2 in zip(list_x1,list_y1,list_x2,list_y2):
            phi_1 = angle(x1,y1,x2,y2)
            
            if phi- math.radians(10) < phi_1 < phi+ math.radians(10):
                line_1_x1.append(x1)
                line_1_y1.append(y1)
                line_1_x2.append(x2)
                line_1_y2.append(y2)
            else:
                line_2_x1.append(x1)
                line_2_y1.append(y1)
                line_2_x2.append(x2)
                line_2_y2.append(y2)
        for x1,x2,y1,y2 in zip(line_1_x1,line_1_x2,line_1_y1,line_1_y2):
        #print(x1,x2,y1,y2)
            a = distance(x1,x2,y1,y2)
            length_xy = round(a,4)
            length =[]
            length.append(length_xy)
            cv2.line(img_3ch_1,(x1,y1),(x2,y2),(0,0,255),1)
        for x1,x2,y1,y2 in zip(line_1_x1,line_1_x2,line_1_y1,line_1_y2):
            b = distance(x1,x2,y1,y2)
            l = round(b,4)
            
            if  l == max(length):
                x1_dst1 = x1
                y1_dst1 = y1
                x2_dst1 = x2
                y2_dst1 = y2
                #print(x1_dst,y1_dst,x2_dst,y2_dst)
                #cv2.line(img_3ch_2,(x1_dst1,y1_dst1),(x2_dst1,y2_dst1),(0,0,255),3)
            
        for x1,x2,y1,y2 in zip(line_2_x1,line_2_x2,line_2_y1,line_2_y2):
            c = distance(x1,x2,y1,y2)
            length_xy = round(c,4) 
            length = []
            length.append(length_xy)
            
            cv2.line(img_3ch_1,(x1,y1),(x2,y2),(0,255,0),1)
        for x1,x2,y1,y2 in zip(line_2_x1,line_2_x2,line_2_y1,line_2_y2):
            d = distance(x1,x2,y1,y2)
            l = round(d,4) 
            if l == max(length):
                x1_dst2 = x1
                y1_dst2 = y1
                x2_dst2 = x2
                y2_dst2 = y2
                #print(x1_dst,y1_dst,x2_dst,y2_dst)
                #cv2.line(img_3ch_2,(x1_dst2,y1_dst2),(x2_dst2,y2_dst2),(0,255,0),3)
        
        #print(i,x1_dst1,y1_dst1,x2_dst1,y2_dst1,x1_dst2,y1_dst2,x2_dst2,y2_dst2)
        
        [a1,b1] = linesolve_inv(x1_dst1,y1_dst1,x2_dst1,y2_dst1)
        
        [a2,b2] = linesolve_inv(x1_dst2,y1_dst2,x2_dst2,y2_dst2)
        
        [x_c,y_c] = cs_solve(a1,b1,a2,b2)
        #print(int(x_c),int(y_c))
        #cv2.circle(img_3ch_2,(int(x_c),int(y_c)),10,(255,0,0),-1)
        
        line1_x_far = []
        line1_y_far = []
        line1_x_near = []
        line1_y_near = []
        for x1,x2,y1,y2 in zip(line_1_x1,line_1_x2,line_1_y1,line_1_y2):
            len1_1 = distance(x1,y1,x_c,y_c)
            len1_2 = distance(x2,y2,x_c,y_c)
            if len1_1 > len1_2:
                line1_x_far.append(x1)
                line1_y_far.append(y1)
                line1_x_near.append(x2)
                line1_y_near.append(y2)
            else:
                line1_x_far.append(x2)
                line1_y_far.append(y2)
                line1_x_near.append(x1)
                line1_y_near.append(y1)
            x1_dst1 = int(list_ave(line1_x_far))
            y1_dst1 = int(list_ave(line1_y_far))
            x2_dst1 = int(list_ave(line1_x_near))
            y2_dst1 = int(list_ave(line1_y_near))
        cv2.line(img_3ch_2,(x1_dst1,y1_dst1),(x2_dst1,y2_dst1),(0,0,255),3)
        
        
        
        
        line2_x_far = []
        line2_y_far = []
        line2_x_near = []
        line2_y_near = []
        
        for x1,x2,y1,y2 in zip(line_2_x1,line_2_x2,line_2_y1,line_2_y2):
            len2_1 = distance(x1,y1,x_c,y_c)
            len2_2 = distance(x2,y2,x_c,y_c)
            if len2_1 > len2_2:
                line2_x_far.append(x1)
                line2_y_far.append(y1)
                line2_x_near.append(x2)
                line2_y_near.append(y2)
            else:
                line2_x_far.append(x2)
                line2_y_far.append(y2)
                line2_x_near.append(x1)
                line2_y_near.append(y1)
            x1_dst2 = int(list_ave(line2_x_far))
            y1_dst2 = int(list_ave(line2_y_far))
            x2_dst2 = int(list_ave(line2_x_near))
            y2_dst2 = int(list_ave(line2_y_near))
            
        cv2.line(img_3ch_2,(x1_dst2,y1_dst2),(x2_dst2,y2_dst2),(0,255,0),3)

        [a1,b1] = linesolve_inv(x1_dst1,y1_dst1,x2_dst1,y2_dst1)
        
        [a2,b2] = linesolve_inv(x1_dst2,y1_dst2,x2_dst2,y2_dst2)
        
        [x_c,y_c] = cs_solve(a1,b1,a2,b2)
        
        cv2.circle(img_3ch_2,(int(x_c),int(y_c)),10,(255,0,0),-1)
        
        [unitvec_x_1,unitvec_y_1,unitvec_x_2,unitvec_y_2] = unitvec(x1_dst1,y1_dst1,x2_dst1,y2_dst1,x1_dst2,y1_dst2,x2_dst2,y2_dst2,x_c,y_c) 
        
        linevalue_1pl = []
        linevalue_1mn = []
        linevalue_2pl = []
        linevalue_2mn = []
        axis_x = []
        j = 1
        while j < 101:
            
            if 0<int(y_c+j*unitvec_y_1)<1023 and 0<int(x_c+j*unitvec_x_1)<1023 and 0<int(y_c-j*unitvec_y_1)<1023 and 0<int(x_c-j*unitvec_x_1)<1023 and 0<int(y_c+j*unitvec_y_2)<1023 and 0<int(x_c+j*unitvec_x_2)<1023 and 0<int(y_c-j*unitvec_y_2)<1023 and 0<int(x_c-j*unitvec_x_2)<1023 :
                linevalue_1pl.append(img[int(y_c+j*unitvec_y_1),int(x_c+j*unitvec_x_1)])
                linevalue_1mn.append(img[int(y_c-j*unitvec_y_1),int(x_c-j*unitvec_x_1)])
                linevalue_2pl.append(img[int(y_c+j*unitvec_y_2),int(x_c+j*unitvec_x_2)])
                linevalue_2mn.append(img[int(y_c-j*unitvec_y_2),int(x_c-j*unitvec_x_2)])
                axis_x.append(j)
                j = j+1
            else:
                linevalue_1pl.append(0)
                linevalue_1mn.append(0)
                linevalue_2pl.append(0)
                linevalue_2mn.append(0)
                axis_x.append(j)
                j = j+1

                continue
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.scatter(linevalue_1x,linevalue_1y,1,c='r')
        ax1.scatter(int(x_c),0,30)
        ax1.set_title("red")
        
        ax1.set_ylabel('brightness')
        ax2 = fig.add_subplot(2,1,2)
        ax2.scatter(linevalue_2x,linevalue_2y,1,c='g')
        ax2.scatter(int(x_c),0,30)
        ax2.set_title("green")
        ax2.set_xlabel('ch along with line')
        ax2.set_ylabel('brightness')
        plt.show()
        '''
        n = len(linevalue_1mn)-linevalue_1mn.count(0)
        g = len(linevalue_2mn)-linevalue_2mn.count(0)
        res = []
        if n>g:
            x_he = x_c-n*unitvec_x_1
            y_he = y_c-n*unitvec_y_1
            x_h3l = x_c-g*unitvec_x_2
            y_h3l = y_c-g*unitvec_y_2
            res.append('red')
        else:
            x_he = x_c-g*unitvec_x_2
            y_he = y_c-g*unitvec_y_2
            x_h3l = x_c-n*unitvec_x_1
            y_h3l = y_c-n*unitvec_y_1
            res.append('green')
        #print(a2,b2)
        #print(np.linalg.solve(a2, b2))

        coordinate["x1_dst1"] = x1_dst1
        coordinate["y1_dst1"] = y1_dst1
        coordinate["x2_dst1"] = x2_dst1
        coordinate["y2_dst1"] = y2_dst1
        coordinate["x1_dst2"] = x1_dst2
        coordinate["y1_dst2"] = y1_dst2
        coordinate["x2_dst2"] = x2_dst2
        coordinate["y2_dst2"] = y2_dst2
        
        coordinate["x_c"] = int(x_c)
        coordinate["y_c"] = int(y_c)
        coordinate["x_he"] = int(x_he)
        coordinate["y_he"] = int(y_he)
        coordinate["x_h3l"] = int(x_h3l)
        coordinate["y_h3l"] = int(y_h3l)
        list_coordinate.append(coordinate)
        print(coordinate['x_h3l'])
        #print(a1,b1,a2,b2)
        writer.writerow([int(x_c),int(y_c),int(x_he),int(y_he)])    
        cv2.imwrite(new_dir_path_hough + str(i) + '.png', img_3ch)
        cv2.imwrite(new_dir_path_line_sep + str(i) + '.png', img_3ch_1)
        cv2.imwrite(new_dir_path_line_sep_1line + str(i) + '.png', img_3ch_2)
        #fig.savefig(new_dir_path_graph_judge + str(i) +'.png')
        i = i+1

    f.close

    with open('coordinate.pickle', 'wb') as f:
        pickle.dump(list_coordinate, f)
