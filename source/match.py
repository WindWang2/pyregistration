import numpy as np
import cv2
from matplotlib import pyplot as plt
def GetCorrctMatchRate(img1, img2):
	MIN_MATCH_COUNT = 1
	# img1 = cv2.imread('box.png', 0)
	# img2 = cv2.imread('box_in_scene.png', 0)

	img1 = cv2.imread(img1, 0)
	img2 = cv2.imread(img2, 0)
	# Initiate SIFT detector
	sif = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sif.detectAndCompute(img1, None)
	kp2, des2 = sif.detectAndCompute(img2, None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params,search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
		des_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
		# Ransac Filter
		M, mask = cv2.findHomography(src_pts,des_pts,cv2.RANSAC,5.0)
		kp_pairs = [kpp for kpp, flag in zip(good, mask) if flag]
	else:
		print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
		# matchesMask = None

	# Calculate the Corrects Matches
	correctPairsCounts = 0
	for tmp in kp_pairs:
		src_ptx,src_pty = kp1[tmp.queryIdx].pt
		des_ptx,des_pty = kp2[tmp.trainIdx].pt

		if(abs(src_ptx-des_ptx) < 1.5 and abs(src_pty-des_pty) < 1.5):
			correctPairsCounts += 1

	return correctPairsCounts/float(len(kp_pairs))
	# Plot for test
	# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
	#                    singlePointColor = None,
	#                    flags = 2)

	# img3 = cv2.drawMatches(img1,kp1,img2,kp2,kp_pairs,None,**draw_params)

	# plt.imshow(img3, 'gray'),plt.show()
if __name__ == '__main__':
	sarPath = '/Users/kevin/Desktop/exp-5-21/country_jpeg/'
	result = []
	txt = open(sarPath+'CorrectsMatch.txt', 'w')
	for i in range(100):
		img = str(i)+'.jpeg'
		re = GetCorrctMatchRate(sarPath+'0.jpeg',sarPath+img)*100
		txt.write('{:.4f}\n'.format(re))
		result.append(re)
		print(i)
	txt.close()
	print result

