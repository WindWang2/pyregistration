#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
#from scipy import misc
import gdal
#from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage import filter
import numpy as np
from numpy import zeros
from numpy import array
from numpy import log2
from numpy import double
from numpy.lib.function_base import histogram
#from numpy import argmax
def CalMI(a,b):

    reference = array(a)
    query = array(b)
    L=2
    miMat = zeros((L,L))
    reference.shape = -1
    query.shape = -1
    miImg = zip(reference.tolist(),query.tolist())
    #print(miImg)
    for m in miImg:
        miMat[m] = miMat[m]+1
    miMat = array(miMat) / double(reference.size)
    refHist,temp = histogram(reference,2,range=(0,2))
    queHist,temp = histogram(query,2,range=(0,2))
    refHist = refHist / double(reference.size)
    queHist = queHist / double(query.size)
    r=-refHist*log2(refHist+0.000000000000000000000000000001)
    q=-queHist*log2(queHist+0.000000000000000000000000000001)
    r[refHist==0]=0
    q[queHist==0]=0
    r = sum(r)
    q = sum(q)
    refHist.shape = refHist.size,1
    rq = (refHist*queHist)
    MI = miMat*log2((miMat)/(rq+0.000000000000000000000001)+0.000000000000000000000001)
    MI[miMat == 0]=0
    MI=sum(MI)
    NMI = 2*MI/(r+q)
    #print(r,q,MI)
    return NMI
# Generate noisy image of a square
# im = np.zeros((128, 128))
# im[32:-32, 32:-32] = 1

# im = misc.imread('sar.tif')
sards = gdal.Open('SAR.tif')
imsar = sards.ReadAsArray()
imsar = np.double(imsar)
imsar_ag = np.zeros((imsar.shape),dtype = float)

opds = gdal.Open('op.tif')
imop=opds.ReadAsArray()
imop = np.double(imop)
imop_age = np.zeros((imop.shape),dtype=float)


#Get Harris Point from optical Image
ds = gdal.Open('op.tif')
image = ds.ReadAsArray()
#coords = corner_peaks(corner_harris(image), min_distance=9)
#poss = zip(coords[:,0],coords[:,1])
# im = im/np.max(im)
# im = ndimage.rotate(im, 15, mode='constant')
#im = ndimage.gaussian_filter(im, 4)
# im += 0.2 * np.random.random(im.shape)

# Compute the Canny filter for two values of sigma
edges1 = filter.canny(imop,sigma = 5)
edges2 = filter.canny(imsar,sigma= 5)



filter.gaussian_filter()
sarX,sarY = edges2.shape
#Find Image Match Point
ws = 11
ij=[]
# for pos in poss[10:25]:
#     x,y = pos
#     a = edges1[x-(ws/2):x+(ws/2+1),y-(ws/2):y+(ws/2+1)]
#     tempNmi = []
#     tempij = []
#     over = False
#     for i in range(sarx-11):
#         for j in range(sary-11):
#             b = edges2[i:i+ws,j:j+ws]
#             tempNmi.append(CalMI(a,b))
#             tempij.append((i+ws/2,j+ws/2))
#     u = np.argmax(np.array(tempNmi))
#     ij.append(tempij[u])
# display results
#fig, (ax1, ax2)  = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

# ax1.imshow(imop, cmap=plt.cm.gray)
# ax1.axis('off')
# ax1.set_title('noisy image', fontsize=20)

#ax2.imshow(edges1, cmap=plt.cm.gray)
#ax2.axis('off')
#ax2.set_title('Op, $\sigma=5$', fontsize=20)
# ax2.plot(poss[1][0],ij[1][1],'+r',markersize = 15)

#ax1.imshow(edges2, cmap=plt.cm.gray)
#ax1.axis('off')
#ax1.set_title('Sar, $\sigma=5$', fontsize=20)
# ax3.plot(ij[0][0],ij[0][1],'+r',markersize = 15)

#fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0.02, left=0.02, right=0.98)


#plt.show()
# plt.savefig('sar_op_edges5.png',dpi = 500)
