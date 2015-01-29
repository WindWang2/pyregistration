# -*- coding: utf-8 -*-
import matplotlib.pylot as plt
import gdal
from skimage import filter
import numpy as np
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
    ---------------------------
    a: data
    b: data
    """

    return np.abs(np.corrcoef(np.reshape(a,(1,-1)),np.reshape(b,(1,-1)))[0,1])


def GetRectBuffer(edge,width):
    """
    Generate the Buffer With edge
    ----------------------------
    edge: the binary edge image
    width: Buffer's width
    """
    if(width % 2 == 0):
        print("The Width Must be odd")
        return 0
    buf = edge.copy()

    # # XXX: Test
    # buf = np.ones((10,10))
    # width = 5

    imgH,imgW = buf.shape

    halfwidth = width/2

    # Expand buf with zeros
    # DirY bottom
    imgBufferY= np.zeros((halfwidth,imgW))
    buf = np.concatenate((buf,imgBufferY))
    # DirY up
    buf = np.concatenate((imgBufferY,buf))
    #DirX Left
    imgBufferX= np.zeros((imgH+halfwidth*2,halfwidth))
    buf = np.concatenate((buf,imgBufferX),axis = 1)
    #DirX Right
    buf = np.concatenate((imgBufferX,buf),axis = 1)


    def __bufferForPoint(x,y):
        """
        Get Rect Area for point
        ----------------------------
        x,y for (x,y) in image
        """
        # i for x, and j for y
        # FIXME: The speed of loop is slow
        for i in range(width):
            for j in range(width):
                buf[i+x,j+y] = 1

    for i in range(imgW):
        for j in range(imgH):
            if(edge[j,i] == 1):
                __bufferForPoint(i,j)

    return buf[halfwidth:(halfwidth+imgH),halfwidth:(halfwidth+imgW)]

def ReadData(file):
    """
    Use GDAL to Read file as Numpy array
    ------------------------------------
    file: The file path
    """

    ds = gdal.Open(file)
    return ds.ReadAsArray()

def GetEdgeByData(data,s):
    """
    Use Skimage's filter canny to get canny edge
    --------------------------------------------
    data: Image data
    sigma: Gaussion Blur' Sigma(Scale)
    """

    return filter.canny(data,sigma = s)

def GetBlurData(data,sigma):
    """
    Use SKimage's Gaussian filter to get Blur image for generate main edge using edges
    ------------------------------------
    data: Image Data
    sigma: Gaussian Blur's Sigma(scale)
    """
    return filter.gaussian_filter(data,sigma)

def GetBufferEdge(data,buffer):
    """
    Use the Buffer to Get Edge
    -----------------------------------
    data: blur image data
    buffer: edge Buffer data
    """
    data[buffer==0]=0
    return data

def readRefPoints(ptFile):
    """
    Read the Points from ptFile
    ----------------------------------
    ptFile: Points file. Contain column X and column Y
    """
    return np.loadtxt(ptFile)

#def
def FindSamePoint(pts,searchWidth,ws,refImage,desImage):
    """
    Find the same points as pts in desImage
    -----------------------------------------------
    pts: points in refImage
    searchWidth: Search width by pixels
    ws: Window Size
    refImage: Reference Image
    desImage: Image for looking points
    """
    # m for the number of pts
    m,n=pts.shape
    desPts=np.zeros(pts.shape)

    for k in range(m):
        x = pts[k,0]
        y = pts[k,1]
        halfSearchWid = searchWidth/2
        halfWs = ws/2
        temp=0
        tempi=0
        tempj=0
        for i in range(-halfSearchWid,halfSearchWid+1):
            for j in range(-halfSearchWid,halfSearchWid+1):
                re = CalCC(refImage[(y-halfWs):(y+halfWs+1),(x-halfWs):(x+halfWs+1)], \
                      desImage[(j+y-halfWs):(j+y+halfWs+1),(i+x-halfWs):(i+x+halfWs+1)])
                if(re>temp):
                    temp = re
                    tempi = i
                    tempj = j
        desPts[k,0]=x+tempi
        desPts[k,1]=x+tempj
    return desPts
