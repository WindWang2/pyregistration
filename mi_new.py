# -*- coding: utf-8 -*-
import matplotlib.pylot as plt
import gdal
from skimage import filter
import numpy as np
# Add the new line
# Add the second line
from numpy.lib.function_base import histogram

def CalMI(a,b):
    """
    Calculate the MI
    """
    reference = np.array(a)
    query = np.array(b)
    L=255
    miMat = np.zeros((L,L))
    reference.shape = -1
    query.shape = -1
    miImg = zip(reference.tolist(),query.tolist())
    #print(miImg)
    for m in miImg:
        miMat[m] = miMat[m]+1
    miMat = np.array(miMat) / np.double(reference.size)
    refHist,temp = histogram(reference,2,range=(0,2))
    queHist,temp = histogram(query,2,range=(0,2))
    refHist = np.refHist / np.double(reference.size)
    queHist = np.queHist / np.double(query.size)
    r=-refHist*np.log2(refHist+0.000000000000000000000000000001)
    q=-queHist*np.log2(queHist+0.000000000000000000000000000001)
    r[refHist==0]=0
    q[queHist==0]=0
    r = sum(r)
    q = sum(q)
    refHist.shape = refHist.size,1
    rq = (refHist*queHist)
    MI = miMat*np.log2((miMat)/(rq+0.000000000000000000000001)+0.000000000000000000000001)
    MI[miMat == 0]=0
    MI=sum(MI)
    NMI = 2*MI/(r+q)
    #print(r,q,MI)
    return NMI

def CalCC(a,b):
    """

    Calculate the CC
    """
    CC = 0.0
    return CC


def GetRectBuffer(edge,width):
    """
    Generate the Buffer With edge
    ----------------------------
    edge: the binary edge image
    width: Buffer's width
    """
    return edge
