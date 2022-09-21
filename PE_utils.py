# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:35:41 2022

@author: Abel
"""
from tensorflow.keras import backend as K
import numpy as np

#%%
expand_type=1
num_terms = 2
#funcType = 'ln(1+x)'
#funcType = 'sin(x)'
funcType = 'arctan(x)'
#%%
def prog_expen(x):     
    #new_x = np.zeros((x.shape[0], x.shape[1],x.shape[2],x.shape[3]*num_terms))
    print("Progressive Expansion of the input")
    print(x.shape)
    num=x.shape[0]
    dim_x=x.shape[1]
    dim_y=x.shape[2]
    bands=x.shape[3]
    new_x = np.zeros((x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]*num_terms))
    new_x = new_x.astype('float32') 
    x=np.reshape(x,(num*dim_x*dim_y,bands))

    nn = 0
    for i in range(x.shape[1]):
        col_d = x[:,i].ravel()
        new_x[:,nn] = col_d
        if num_terms > 0:
            if expand_type == 1:
                for od in range(1,num_terms):
                    if funcType  == 'linear':
                       new_x[:,nn+od] = col_d
                    elif funcType == 'sin(x)': # sin(x)
                       new_x[:,nn+od] = new_x[:,nn+od-1] + (((-1)**od)/np.math.factorial(2*od+1))*(col_d**(2*od+1))  
                    elif funcType == 'ln(1+x)':  # log(1+x)
                       new_x[:,nn+od] = new_x[:,nn+od-1] + ((-1)**(od+1+1))*(col_d**(od+1))/(od+1)  # require |x|<1. 
                    elif funcType == 'arctan(x)':  # 
                       new_x[:,nn+od] = new_x[:,nn+od-1] + ((-1)**(od))*(col_d**(2*od+1)/(2*od+1))
                nn = nn + num_terms
            else:
                for od in range(1,num_terms):
                    if funcType  == 'linear':
                       new_x[:,nn+od] = col_d
                    elif funcType == 'sin(x)': # sin(x)
                       new_x[:,nn+od] = (((-1)**od)/np.math.factorial(2*od+1))*(col_d**(2*od+1))  
                    elif funcType == 'ln(1+x)':  # log(1+x)
                       new_x[:,nn+od] = ((-1)**(od+1+1))*(col_d**(od+1))/(od+1)  # require |x|<1. 
                    elif funcType == 'arctan(x)':  # 
                       new_x[:,nn+od] = ((-1)**(od))*(col_d**(2*od+1)/(2*od+1))
                nn = nn + num_terms                
    
    x1 = new_x[:,0::3]
    x2 = new_x[:,1::3]
    x3 = new_x[:,2::3]
    cat_x = np.hstack((x1,x2,x3))
    cat_x=np.reshape(cat_x,(num,dim_x,dim_y,bands*num_terms))
    return cat_x

#%%
def prog_expen_conv1(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
    else:
        p1 = x
        p2 = -((x**2)/2)     
    new_x = K.concatenate([p1, p2], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv2(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
    else:
        p1 = x
        p2 = -((x**2)/2)
    new_x = K.concatenate([p1, p2], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv3(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        #p3 = x - ((x**2)/2)
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
    new_x = K.concatenate([p1, p2, p3], axis=len(x.shape)-1)
#    print(new_x)
    return new_x




def prog_expen_conv4(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
    new_x = K.concatenate([p1, p2, p3, p4], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv5(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
    new_x = K.concatenate([p1, p2, p3, p4, p5], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv6(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
        p6 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6)
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
        p6 = - ((x**6)/6)
    new_x = K.concatenate([p1, p2, p3, p4, p5, p6], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv7(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
        p6 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6)
        p7 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7)
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
        p6 = - ((x**6)/6)
        p7 = ((x**7)/7)
    new_x = K.concatenate([p1, p2, p3, p4, p5, p6, p7], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv8(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
        p6 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6)
        p7 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7)
        p8 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) 
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
        p6 = - ((x**6)/6)
        p7 = ((x**7)/7)
        p8 = - ((x**8)/8) 
    new_x = K.concatenate([p1, p2, p3, p4, p5, p6, p7, p8], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv9(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
        p6 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6)
        p7 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7)
        p8 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) 
        p9 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) + ((x**9)/9)
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
        p6 = - ((x**6)/6)
        p7 = ((x**7)/7)
        p8 = - ((x**8)/8) 
        p9 = ((x**9)/9)
    new_x = K.concatenate([p1, p2, p3, p4, p5, p6, p7, p8, p9], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv10(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
        p6 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6)
        p7 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7)
        p8 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) 
        p9 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) + ((x**9)/9)
        p10 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) + ((x**9)/9)- ((x**10)/10)
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
        p6 = - ((x**6)/6)
        p7 = ((x**7)/7)
        p8 = - ((x**8)/8) 
        p9 = ((x**9)/9)
        p10 = - ((x**10)/10)
    new_x = K.concatenate([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10], axis=len(x.shape)-1)
#    print(new_x)
    return new_x
