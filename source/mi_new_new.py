# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import gdal
from skimage import filter
import numpy as np
from numpy.lib.function_base import histogram
import scipy.io as sci
import skimage


def CalMI(a, b):
    """
    Calculate the MI
    """
    reference = np.array(a)
    query = np.array(b)
    L = 256
    miMat = np.zeros((L, L))
    reference.shape = -1
    query.shape = -1
    miImg = zip(reference.tolist(), query.tolist())
    # print(miImg)
    for m in miImg:
        miMat[m] = miMat[m]+1
    miMat = np.array(miMat) / np.double(reference.size)
    refHist, temp = histogram(reference, 256, range=(0, 256))
    queHist, temp = histogram(query, 256, range=(0, 256))
    refHist = refHist / np.double(reference.size)
    queHist = queHist / np.double(query.size)
    r = -refHist*np.log2(refHist+0.000000000000000000000000000001)
    q = -queHist*np.log2(queHist+0.000000000000000000000000000001)
    r[refHist == 0] = 0
    q[queHist == 0] = 0
    r = np.sum(r)
    q = np.sum(q)
    refHist.shape = refHist.size, 1
    rq = (refHist*queHist)
    MI = miMat * np.log2(
        (miMat) / (rq + 0.00000000000001) + 0.000000000000001)
    MI[miMat == 0] = 0
    MI = np.sum(MI)
    NMI = 2*MI/(r+q)
    # print(r,q,MI)
    return NMI


def CalCC(a, b):
    """
    Calculate the CC
    ---------------------------
    a: data
    b: data
    """
    if a.ndim == 3:
        m, n, band = a.shape
        sum = 0.0
        # for i in range(band):
        #    sum+=np.abs(np.corrcoef(np.reshape(a[:,:,i],(1,-1)),np.reshape(b[:,:,i],(1,-1)))[0,1])
        sum = np.abs(np.corrcoef(
            np.reshape(
                a[:, :, :], (1, -1)), np.reshape(b[:, :, :], (1, -1)))[0, 1])
        return sum
    else:
        return np.abs(np.corrcoef(
            np.reshape(a, (1, -1)), np.reshape(b, (1, -1)))[0, 1])


def GetRectBuffer(edge, width):
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

    imgH, imgW = buf.shape

    halfwidth = width/2

    # Expand buf with zeros
    # DirY bottom
    imgBufferY = np.zeros((halfwidth, imgW))
    buf = np.concatenate((buf, imgBufferY))
    # DirY up
    buf = np.concatenate((imgBufferY, buf))
    # DirX Left
    imgBufferX = np.zeros((imgH + halfwidth * 2, halfwidth))
    buf = np.concatenate((buf, imgBufferX), axis=1)
    # D irX Right
    buf = np.concatenate((imgBufferX, buf), axis=1)

    def __bufferForPoint(x, y):
        """
        Get Rect Area for point
        ----------------------------
        x,y for (x,y) in image
        """
        # i for x, and j for y
        # FIXME: The speed of loop is slow
        for i in range(width):
            for j in range(width):
                buf[j+y, i+x] = 1

    for i in range(imgW):
        for j in range(imgH):
            if(edge[j, i] == 1):
                __bufferForPoint(i, j)

    return buf[halfwidth:(halfwidth + imgH), halfwidth:(halfwidth + imgW)]


def ReadData(file):
    """
    Use GDAL to Read file as Numpy array
    ------------------------------------
    file: The file path
    """

    ds = gdal.Open(file)
    return ds.ReadAsArray()


def GetEdgeByData(data, s):
    """
    Use Skimage's filter canny to get canny edge
    --------------------------------------------
    data: Image data
    sigma: Gaussion Blur' Sigma(Scale)
    """

    return filter.canny(data, sigma=s)


def GetBlurData(data, sigma):
    """
    Use SKimage's Gaussian filter to get Blur image
    for generate main edge using edges
    ------------------------------------
    data: Image Data
    sigma: Gaussian Blur's Sigma(scale)
    """
    return filter.gaussian_filter(data, sigma)


def GetBufferEdge(data, buffer):
    """
    Use the Buffer to Get Edge
    -----------------------------------
    data: blur image data
    buffer: edge Buffer data
    """
    data[buffer == 0] = 0
    return data


def readRefPoints(ptFile):
    """
    Read the Points from ptFile
    ----------------------------------
    ptFile: Points file. Contain column X and column Y
    """
    return np.loadtxt(ptFile)


def FindSamePoint(pts, searchWidth, ws, refImage, desImage):
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
    m, n = pts.shape
    desPts = np.zeros(pts.shape)

    for k in range(m):
        x = pts[k, 0]
        y = pts[k, 1]
        halfSearchWid = searchWidth/2
        halfWs = ws/2
        temp = 0
        tempi = 0
        tempj = 0
        test = []
        print(k)
        for i in range(-halfSearchWid, halfSearchWid+1):
            for j in range(-halfSearchWid, halfSearchWid+1):
                # re = CalMI(refImage[(y-halfWs):(y+halfWs+1),(x-halfWs):(x+halfWs+1)], \
                #       desImage[(j+y-halfWs):(j+y+halfWs+1),(i+x-halfWs):(i+x+halfWs+1)])
                a = refImage[(y-halfWs):(y+halfWs+1), (x-halfWs):(x+halfWs+1)]
                b = desImage[(j+y-halfWs):(j+y+halfWs+1), (i+x-halfWs):(i+x+halfWs+1)]
                re = CalCC(a, b)
                test.append(re)
                if(re > temp):
                    temp = re
                    tempi = i
                    tempj = j
        desPts[k, 0] = x+tempi
        desPts[k, 1] = y+tempj
        np.savetxt(
            '/Users/kevin/Desktop/'+str(k)+'.pts', np.array(test), fmt='%.4f')
    return desPts


def plotwithpixels(img, pts, outFile, textcolor):
    """
    plot the image with pts
    ----------------------------
    img: image
    pts: point([x,y])
    outFile: the path of outImageFile
    color: TEXT Color
    """
    plt.imshow(img, plt.cm.gray)
    plt.plot(pts[:, 0], pts[:, 1], 'r+', markersize=15)

    h, w = img.shape

    n, temp = pts.shape

    for i in range(n):
        plt.text(
            pts[i, 0]+10, pts[i, 1]-10, str(i+1), fontsize=15, color=textcolor)

    plt.xlim((0, w))
    plt.ylim((h, 0))

    plt.axis('off')
    plt.savefig(outFile, dpi=300)
    plt.close()


def GetGradient(img):
    """

    """
    a, b = np.gradient(np.double(img))
    return np.sqrt((a**2+b**2)/2)

if __name__ == '__main__':
    """
    test function
    """
    workspace = '/Users/kevin/work/gPb/grouping/'
    ptsfile = workspace+'Stest2gcp3.pts'
    pts_all = np.loadtxt(ptsfile)
    pts = pts_all[:, 0:2]
    ref_pts = pts_all[:, 2:]
    # op_image
    opfile = workspace+'op.tif'
    # sar_image
    sarfile = workspace+'sar.tif'

    opimg = ReadData(opfile)
    sarimg = ReadData(sarfile)
    #    pts = readRefPoints(oppts)
    #    #sigma
    # searchwidth
    searchwidth = 50
    # windows size
    ws = 39
    # repts =
    # FindSamePoint(pts,searchwidth,ws,GetGradient(opimg),GetGradient(sarimg))
    # repts = FindSamePoint(pts,searchwidth,ws,opbufedge,sarbufedge)

    tempSAR = sci.loadmat(workspace+'gpb_sar.mat')
    tempOP = sci.loadmat(workspace+'op_gpb.mat')
    maxOpgPb = np.max(tempOP['gPb_orient'], axis=2)
    maxSargPb = np.max(tempSAR['gPb_orient'], axis=2)
    repts = FindSamePoint(
        pts, searchwidth, ws, maxOpgPb, maxSargPb)
    firstOpgPb = tempOP['gPb_orient'][:, :, 0]
    firstSargPb = tempSAR['gPb_orient'][:, :, 0]
    repts = FindSamePoint(
        pts, searchwidth, ws, firstOpgPb, firstSargPb)
    # XXX: Test in Origin Image
    # repts = FindSamePoint(pts,searchwidth,ws,opimg,sarimg)

    # plt.imsave(workspace+'opbufedge.tif',opbufedge,cmap=plt.cm.gray)
    # plt.imsave(workspace+'sarbufedge.tif',sarbufedge,cmap = plt.cm.gray)

    plotwithpixels(opimg, pts, workspace+'opre.tif', 'yellow')
    plotwithpixels(sarimg, repts, workspace+'sarre.tif', 'red')

    np.savetxt(workspace+'test.txt', repts, fmt='%.0d')
    print(repts)
