import numpy as np
import itertools
import xml.etree.ElementTree as ET
PI = np.pi

def conv_grad_b (grad_array, data_array):

    GAMMA = return_gamma(data_array[0])
    delta = data_array[0]['acqus']['P'][30]*2 / 1E6 # For ledbpgp2s
    DELTA = data_array[0]['acqus']['D'][20] - data_array[0]['acqus']['D'][16] / 2
    calcB = (2 * PI * GAMMA * delta * np.array(grad_array))**2 * (DELTA - delta/3 ) * 1E4

    return calcB

def return_gamma(dic):

    if dic['acqus']['NUC1'] == '1H':

        GAMMA = 4258 

    elif dic['acqus']['NUC1'] != '1H':

        GAMMA = dic['acqus']['BF1']/ dic['proc2s']['SF'] * 4258 

    return GAMMA

def read_qvalues(path, firstexpno = None, lastexpno = None):

    if firstexpno is None:

        tree = ET.parse(path + '/diff.xml') 
        root = tree.getroot()
        qvalues = [values.text.split(' ') for values in root.iter('List') ]
        qvalues = list(map(float,list(itertools.chain(*qvalues))))
        qvalues = qvalues[2::4]

    else:

        qvalues = []

        for expno in range(firstexpno, lastexpno):
            tree = ET.parse(path + str(expno) + '/diff.xml')
            root = tree.getroot()
            q = [values.text.split(' ') for values in root.iter('List') ]
            q = list(map(float,list(itertools.chain(*q))))
            q = q[2::4]
            qvalues.append(q)
        
    return qvalues

