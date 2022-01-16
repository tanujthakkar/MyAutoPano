#!/usr/env/bin python3

# Importing modules
import cv2
import numpy as np
import math
import os
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, corner_peaks
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from Helper import *


class MyAutoPano():

	def __init__(self):
		self.ImageSet = list()
		self.ImageSetGray = list()
		self.ImageSetHarrisCorners = list()
		self.ImageSetShiTomasiCorners = list()
		self.HarrisCorners = list()
		self.ShiTomasiCorners = list()
		self.ImageSetLocalMaxima = list()
		self.ImageSetANMS = list()
		self.ANMSCorners = list()
		self.Features = list()
		self.Matches = list()
		self.Inliers = list()
		self.Homography = list()
		self.ImageSetRefId = None

		# Toggles
		self.Visualize = False

	def createImageSet(self, ImageSet, N=None, height=None, width=None):
		if(N == None):
			N = len(ImageSet)
		[self.ImageSet.append(cv2.imread(ImageSet[img])) for img in range(N)]
		[self.ImageSetGray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in self.ImageSet]
		self.ImageSet = np.array(self.ImageSet)
		self.ImageSetGray = np.float32(np.array(self.ImageSetGray))
		self.ImageSetRefId = int(len(self.ImageSet)/2)

	def computeHarrisCorners(self, Visualize):
		print("Computing Harris Corners...")
		self.ImageSetHarrisCorners = np.copy(self.ImageSet)
		for i in range(len(self.ImageSetGray)):
			img = cv2.cornerHarris(self.ImageSetGray[i], 2, 3, 0.00001) # Computing corner probability using Harris corners
			# img = cv2.normalize(img, None, -1.0, 1.0, cv2.NORM_MINMAX) # Normalizing
			img[img<0.001*img.max()] = 0
			img = cv2.dilate(img, None) # Dilating to mark corners
			self.ImageSetHarrisCorners[i][img>0.001*img.max()]=[0,0,255] # Marking corners in RGB image
			self.HarrisCorners.append(img)
			if(Visualize):
				# cv2.imshow("", img)
				cv2.imshow("Harris Corners", self.ImageSetHarrisCorners[i])
				cv2.imshow("cimg", self.HarrisCorners[i])
				cv2.waitKey(0)
		self.HarrisCorners = np.float32(np.array(self.HarrisCorners))

	def computeShiTomasiCorners(self, N_best, Visualize):
		print("Computing Shi-Tomasi Corners...")
		self.ImageSetShiTomasiCorners = np.copy(self.ImageSet)
		for img in range(len(self.ImageSetGray)):
			corners = cv2.goodFeaturesToTrack(self.ImageSetGray[img], N_best, 0.01, 10) # Computing corners using the Shi-Tomasi method
			corners = np.int0(corners)
			print("ShiTomasiCorners in Image %d: %d"%(img, len(corners)))

			ShiTomasiCorners = np.zeros(self.ImageSet.shape[1:3])
			for corner in corners: # Marking corners in RGB image
				x,y = corner.ravel()
				ShiTomasiCorners[y,x] = 255
				cv2.circle(self.ImageSetShiTomasiCorners[img],(x,y),2,(0,0,255),-1)

			self.ShiTomasiCorners.append(ShiTomasiCorners)

			if(Visualize):
				cv2.imshow("Shi-Tomasi Corners", self.ImageSetShiTomasiCorners[img])
				cv2.imshow("Corners", self.ShiTomasiCorners[img])
				cv2.waitKey(0)

		self.ShiTomasiCorners = np.array(self.ShiTomasiCorners)
		print(self.ShiTomasiCorners.shape)

	def ANMS(self, ImageSetCorners, N_best, Visualize):
		print("Applying ANMS...")

		self.ImageSetLocalMaxima = np.copy(self.ImageSet)
		self.ImageSetANMS = np.copy(self.ImageSet)
		for img in range(len(ImageSetCorners)):
			ANMSCorners = list()
			local_maximas = peak_local_max(ImageSetCorners[img], min_distance=5)
			local_maximas = np.int0(local_maximas)
			print("local_maximas: %d"%len(local_maximas))

			r = [np.Infinity for i in range(len(local_maximas))]
			ED = 0

			for i in tqdm(range(len(local_maximas))):
				for j in range(len(local_maximas)):
					if(ImageSetCorners[img][local_maximas[j,0],local_maximas[j,1]] > ImageSetCorners[img][local_maximas[i,0],local_maximas[i,1]]):
						ED = math.sqrt((local_maximas[j,0] - local_maximas[i,0])**2 + (local_maximas[j,1] - local_maximas[i,1])**2)
						# print(ED)
					if(ED < r[i]):
						r[i] = ED
				ANMSCorners.append([r[i], local_maximas[i,0], local_maximas[i,1]])

			ANMSCorners = sorted(ANMSCorners, reverse=True)
			ANMSCorners = np.array(ANMSCorners[:N_best])
			print("ANMS Corners: %d"%len(ANMSCorners))
			self.ANMSCorners.append(ANMSCorners)

			if(Visualize):
				for local_maxima in local_maximas: # Marking corners in RGB image
					y,x = local_maxima.ravel()
					cv2.circle(self.ImageSetLocalMaxima[img],(x,y),2,(0,255,0),-1)
					# cv2.circle(self.ImageSetANMS[img],(x,y),2,(0,255,0),-1)

				for i in range(N_best): # Marking corners in RGB image
					cv2.circle(self.ImageSetANMS[img],(int(ANMSCorners[i][2]),int(ANMSCorners[i][1])),2,(0,0,255),-1)

				cv2.imshow("Local Max", self.ImageSetLocalMaxima[img])
				cv2.imshow("ANMS", self.ImageSetANMS[img])
				cv2.waitKey(0)

		self.ANMSCorners = np.array(self.ANMSCorners)
		# print(self.ANMSCorners.shape)

	def featureDescriptor(self, key_points, Visualize):
		print("Retrieving feature patches...")

		for img in range(len(self.ImageSetGray)):
			patch_size = 40
			features = list()
			for point in range(len(key_points[img])):
				patch = np.uint8(np.array(neighbors(self.ImageSetGray[img], 20, int(key_points[img][point][1]), int(key_points[img][point][2]))))
				patch_gauss = cv2.resize(cv2.GaussianBlur(patch, (5,5), 0), None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
				patch_gauss = (patch_gauss - patch_gauss.mean())/patch_gauss.std()
				features.append(patch_gauss.flatten())
				if(Visualize):
					temp = cv2.circle(np.copy(self.ImageSet[img]),(int(key_points[img][point][2]), int(key_points[img][point][1])),2,(0,0,255),-1)
					cv2.imshow("Feature", temp)
					# cv2.imshow("ANMS", self.ImageSetANMS[img])
					cv2.imshow("Patch", patch)
					cv2.imshow("Patch gauss", patch_gauss)
					cv2.waitKey(0)

			features = np.array(features)
			self.Features.append(features)

		self.Features = np.array(self.Features)

	def featureMatching(self, Visualize):
		print("Matching features...")
		for img in range(len(self.ImageSet)-1):
			
			SSDs = list()
			matches = list()
			features = np.arange(len(self.Features[img])).tolist()
			temp = np.hstack((self.ImageSet[img], self.ImageSet[img+1]))
			for i in tqdm(range(len(self.Features[img]))):
				SSDs.clear()
				for j in features:
					SSDs.append([sum((self.Features[img][i] - self.Features[img+1][j])**2), self.ANMSCorners[img+1][j][1], self.ANMSCorners[img+1][j][2]])

				SSDs = sorted(SSDs)
				# print([self.ANMSCorners[img][i][1], self.ANMSCorners[img][i][2], SSDs[0][1], SSDs[0][2]])
				matches.append([self.ANMSCorners[img][i][1], self.ANMSCorners[img][i][2], SSDs[0][1], SSDs[0][2]])
				# input('q')
				# features.remove(SSDs[0][1])

				if(Visualize):
					# temp = np.hstack((self.ImageSet[img], self.ImageSet[img+1]))
					temp = cv2.circle(temp,(int(self.ANMSCorners[img][i][2]), int(self.ANMSCorners[img][i][1])),2,(0,0,255),-1)
					temp = cv2.circle(temp,(int(SSDs[0][2])+self.ImageSet[img].shape[1], int(SSDs[0][1])),2,(0,0,255),-1)
					temp = cv2.line(temp, (int(self.ANMSCorners[img][i][2]), int(self.ANMSCorners[img][i][1])), (int(SSDs[0][2])+self.ImageSet[img].shape[1], int(SSDs[0][1])), (0,255,0), 1)
					cv2.imshow("1", temp)
					# cv2.imshow("2", temp2)

			print("Matches: %d", len(matches))
			cv2.waitKey(0)

			matches = np.array(matches)
			self.Matches.append(matches)

		self.Matches = np.array(self.Matches)
		print(self.Matches.shape)

	def RANSAC(self, iterations, threshold, Visualize):
		print("Performing RANSAC...")
		for img in range(len(self.ImageSet)-1):

			max_inliers = 0
			best_H = None
			Inliers = list()
			features = np.arange(len(self.Features[img])).tolist()
			for i in tqdm(range(iterations)):

				feature_pairs = np.random.choice(features, 4, replace=False)
				p1 = list()
				p2 = list()
				for j in range(len(feature_pairs)):
					p1.append([self.ANMSCorners[img][self.Matches[img][feature_pairs[j]][0]][1], self.ANMSCorners[img][self.Matches[img][feature_pairs[j]][0]][2]])
					p2.append([self.ANMSCorners[img+1][self.Matches[img][feature_pairs[j]][1]][1], self.ANMSCorners[img+1][self.Matches[img][feature_pairs[j]][1]][2]])

				# p1 = np.array(p1)
				# p2 = np.array(p2)

				H = cv2.getPerspectiveTransform(np.float32(p1), np.float32(p2))
				Hp1 = np.dot(H, np.vstack((self.ANMSCorners[img][:,1], self.ANMSCorners[img][:,2], np.ones([1,len(self.ANMSCorners[img])]))))
				Hp1 = np.array(Hp1/(Hp1[2]+1e-20)).transpose()
				Hp1 = np.delete(Hp1, 2, 1)
				p2_ = list()
				[p2_.append([self.ANMSCorners[img+1][self.Matches[img][x][1]][1], self.ANMSCorners[img+1][self.Matches[img][x][1]][2]]) for x in range(len(self.Matches[img]))]
				p2_ = np.array(p2_)

				SSD = list()
				[SSD.append(sum((p2_[x] - Hp1[x])**2)) for x in range(len(self.Features[img]))]
				SSD = np.array(SSD)
				SSD[SSD <= threshold] = 1
				SSD[SSD > threshold] = 0

				inliers = np.sum(SSD)

				if(inliers > max_inliers):
					max_inliers = inliers
					Inliers = np.where(SSD == 1)
					best_H = H
					print("Inliers: %d"%max_inliers)
					print("H: ", H)
					# temp = np.hstack((self.ImageSet[img], self.ImageSet[img+1]))
					# for i in range(4):
					# 	temp = cv2.circle(temp,(int(p1[i][1]), int(p1[i][0])),2,(0,0,255),-1)
					# 	temp = cv2.circle(temp,(int(p2[i][1])+self.ImageSet[img].shape[1], int(p2[i][0])),2,(0,0,255),-1)
					# 	temp = cv2.line(temp, (int(p1[i][1]), int(p1[i][0])), (int(p2[i][1])+self.ImageSet[img].shape[1], int(p2[i][0])), (0,255,0), 1)
					# 	# print((int(self.ANMSCorners[img][i][1]), int(self.ANMSCorners[img][i][2])), (int(self.ANMSCorners[img+1][self.Matches[img][i][1]][1]), int(self.ANMSCorners[img+1][self.Matches[img][i][1]][2])))
					# cv2.imshow("", temp)
					# cv2.waitKey(0)

			# p1 = list()
			# p2 = list()
			# for i in Inliers[0]:
			# 		p1.append([int(self.ANMSCorners[img][i][1]), int(self.ANMSCorners[img][i][2])])
			# 		p2.append([int(self.ANMSCorners[img+1][self.Matches[img][i][1]][1]), int(self.ANMSCorners[img+1][self.Matches[img][i][1]][2])])
			# print(np.float32(p1[10:14]), np.float32(p2[10:14]))
			# H = cv2.getPerspectiveTransform(np.float32(p1[10:14]), np.float32(p2[10:14]))
			# print("Test", H)
			# best_H = H
			# input('q')

			print("Inliers: %d"%max_inliers)
			print("Homography Matrix: ", best_H)
			if(Visualize):
				temp = np.hstack((self.ImageSet[img], self.ImageSet[img+1]))
				for i in Inliers[0]:
					temp = cv2.circle(temp,(int(self.ANMSCorners[img][i][2]), int(self.ANMSCorners[img][i][1])),2,(0,0,255),-1)
					temp = cv2.circle(temp,(int(self.ANMSCorners[img+1][self.Matches[img][i][1]][2])+self.ImageSet[img].shape[1], int(self.ANMSCorners[img+1][self.Matches[img][i][1]][1])),2,(0,0,255),-1)
					temp = cv2.line(temp, (int(self.ANMSCorners[img][i][2]), int(self.ANMSCorners[img][i][1])), (int(self.ANMSCorners[img+1][self.Matches[img][i][1]][2])+self.ImageSet[img].shape[1], int(self.ANMSCorners[img+1][self.Matches[img][i][1]][1])), (0,255,0), 1)
					# print((int(self.ANMSCorners[img][i][1]), int(self.ANMSCorners[img][i][2])), (int(self.ANMSCorners[img+1][self.Matches[img][i][1]][1]), int(self.ANMSCorners[img+1][self.Matches[img][i][1]][2])))
				cv2.imshow("", temp)
				cv2.waitKey(0)

			self.Inliers.append(np.array(Inliers[0]).reshape((-1,1)))
			self.Homography.append(np.array([best_H]))

		self.Inliers = np.array(self.Inliers, dtype=object)
		self.Homography = np.array(self.Homography)

	def blendImages(self, Visualize):
		print("Blending Images...")
		for img in range(len(self.ImageSet)-1):
			H = 0
			h0, w0 = self.ImageSet[img].shape[:2]
			h1, w1 = self.ImageSet[img+1].shape[:2]

			c0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1, 1, 2)
			c1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

			# print(c0)
			# print(c1)
			print("Homography Matrix", self.Homography[img][0])
			H_ = np.array([[ 1.19304625e+00  ,1.47056751e-01 ,-6.33663861e+01],
			 [ 2.79421187e-02  ,1.22945049e+00 ,-2.91078391e+02],
			 [ 2.77550589e-05  ,5.15827151e-04 , 1.00000000e+00]])
			self.Homography[img][0] = H
			c0_ = cv2.perspectiveTransform(c0, self.Homography[img][0])

			print(c0_)
			# print(c0_.shape)

			points_on_image0_transformed_ = list()
			for p in range(len(c0_)):
				points_on_image0_transformed_.append(c0_[p].ravel())

			points_on_image0_transformed_ = np.array(points_on_image0_transformed_)
			print(points_on_image0_transformed_.shape)
			print(points_on_image0_transformed_)

			x_min, y_min = np.int0(np.min(points_on_image0_transformed_, axis = 0))
			x_max, y_max = np.int0(np.max(points_on_image0_transformed_, axis = 0))
			
			print("min, max")
			print(x_min, y_min)
			print(x_max, y_max)

			H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) # translate

			image0_transformed = cv2.warpPerspective(self.ImageSet[img], np.dot(H_translate, self.Homography[img][0]), (x_max-x_min, y_max-y_min))

			images_stitched = image0_transformed.copy()
			print(images_stitched.shape)
			print("test", -y_min, -y_min+h1, -x_min, -x_min+w1)
			images_stitched[-y_min:-y_min+h1, -x_min: -x_min+w1] = self.ImageSet[img+1]

			indices = np.where(self.ImageSet[img+1] == [0,0,0])
			y = indices[0] + -y_min 
			x = indices[1] + -x_min 

			images_stitched[y,x] = image0_transformed[y,x]

			# images_stitched = image0_transformed
			# images_stitched[-y_min:-y_min+h1, -x_min: -x_min+w1] = self.ImageSet[img+1]

			# for y in range(0, h1):
			# 	for x in range(0, w1):
			# 		if(self.ImageSet[img+1][y,x,:] == [0,0,0]):
			# 			images_stitched[-y_min + y, -x_min + x] = image0_transformed[-y_min + y, -x_min + x]

			# p = np.concatenate((c0_, c1), axis=0)

			# temp = cv2.warpPerspective(self.ImageSet[img], np.linalg.inv(self.Homography[H][0]))
			# temp[0:self.ImageSet[int(len(self.ImageSet)/2)].shape[0], 0:self.ImageSet[int(len(self.ImageSet)/2)].shape[1]] = self.ImageSet[int(len(self.ImageSet)/2)]
			H += 1

			if(Visualize):
				cv2.imshow("IMG", self.ImageSet[img])
				cv2.imshow("Ref", self.ImageSet[img+1])
				cv2.imshow("Transformed", image0_transformed)
				cv2.imshow("Stiched", images_stitched)
				cv2.waitKey(0)

	def stitchImagePairs(self, img0, img1, H):

		image0 = img0.copy()
		image1 = img1.copy()

		#stitch image 0 on image 1
		print("shapes")
		print(image0.shape)
		print(image1.shape)
		
		# H_ = np.array([[ 1.19304625e+00  ,1.47056751e-01 ,-6.33663861e+01],
		# 	 [ 2.79421187e-02  ,1.22945049e+00 ,-2.91078391e+02],
		# 	 [ 2.77550589e-05  ,5.15827151e-04 , 1.00000000e+00]])

		# print(abs(H-H_))

		h0 ,w0 ,_ = image0.shape
		h1 ,w1 ,_ = image1.shape

		points_on_image0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1,1,2)
		# H = np.linalg.inv(H)
		print("Homography Matrix", H)
		points_on_image0_transformed = cv2.perspectiveTransform(points_on_image0, H)
		print("transformed points = ", points_on_image0_transformed)
		points_on_image1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1,1,2)

		points_on_merged_images = np.concatenate((points_on_image0_transformed, points_on_image1), axis = 0)
		points_on_merged_images_ = []

		for p in range(len(points_on_merged_images)):
			points_on_merged_images_.append(points_on_merged_images[p].ravel())

		points_on_merged_images_ = np.array(points_on_merged_images_)

		x_min, y_min = np.int0(np.min(points_on_merged_images_, axis = 0))
		x_max, y_max = np.int0(np.max(points_on_merged_images_, axis = 0))

		print("min, max")
		print(x_min, y_min)
		print(x_max, y_max)

		# overlap_area = cv2.polylines(image1,[np.int32(points_on_image0_transformed)],True,255,3, cv2.LINE_AA) 
		# cv2.imshow("original_image_overlapping.jpg", overlap_area)
		# cv2.waitKey() 
		# cv2.destroyAllWindows()
		H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) # translate

		image0_transformed_and_stitched = cv2.warpPerspective(image0, np.dot(H_translate, H), (x_max-x_min, y_max-y_min))

		#image0_transformed_and_stitched[-y_min:-y_min+h1, -x_min: -x_min+w1] = image1

		images_stitched = image0_transformed_and_stitched.copy()
		images_stitched[-y_min:-y_min+h1, -x_min: -x_min+w1] = image1

		indices = np.where(image1 == [0,0,0])
		y = indices[0] + -y_min 
		x = indices[1] + -x_min 

		images_stitched[y,x] = image0_transformed_and_stitched[y,x]

		cv2.imshow("", image0_transformed_and_stitched)
		cv2.imshow("s", images_stitched)
		cv2.waitKey(0)
		
		return images_stitched

	def test(self, img1, img2):
		sift = cv2.xfeatures2d.SIFT_create()
		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(img1,None)
		kp2, des2 = sift.detectAndCompute(img2,None)

		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1,des2, k=2)

		# Apply ratio test
		good = []
		for m in matches:
			if m[0].distance < 0.5*m[1].distance:
				good.append(m)
		matches = np.asarray(good)

		if len(matches[:,0]) >= 4:
			src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
			dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

		H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
		# H_ = cv2.getPerspectiveTransform(src, dst)

		print(H)

		return H