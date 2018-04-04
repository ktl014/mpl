# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 2017

Pull out standard hand engineered features from SPC images. 

@author: eric
"""
import os
import numpy as np
import cv2
from skimage import morphology, measure, exposure
from skimage.filters import threshold_otsu
from skimage.feature import greycomatrix, greycoprops
from scipy import fftpack, interpolate, ndimage
from math import pi
import glob
import os
import json
import time
import caffe


# Extract features from a single image
def extractFeatures(img):
    assert isinstance(img, (list, tuple, np.ndarray)) # Img must be array type or scalar

    # Define the full feature vector
    X = np.zeros((72,))
    
    # threshold
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = threshold_otsu(gray)
    binary = np.where(gray > thresh, 0., 1.0)
    bw_img1 = morphology.closing(binary, morphology.square(3))
    
    # pad to ensure contour is continuous
    bw_img = np.pad(bw_img1, 1, 'constant')
    
    # Find contours
    contours = measure.find_contours(bw_img, 0.5)
    
    # Select largest contour
    maxLength = -1
    maxContour = []
    for cc in contours:
        if (len(cc) > maxLength):
            maxLength = len(cc)
            maxContour = cc
    
    # Represent contour in fourier space. Make scale invarient by 
    # dividing by DC term. Can make rotation invariant by subtracting 
    # phase of first term        

    # Interpolate to 4096 point contour
    interpX = interpolate.interp1d(range(0, maxLength), maxContour[:,0])
    interpY = interpolate.interp1d(range(0, maxLength), maxContour[:,1])
    newS = np.linspace(0, maxLength-1, 4096)
    cX = interpX(newS)
    cY = interpY(newS)
    cPath = cX +1j*cY
    FdAll = np.fft.fft(cPath)
        
    # Simplify the boundary
    cen = np.fft.fftshift(FdAll)
    
    # take first 10% of fourier coefficents
    cen2 = np.hstack([np.zeros(1843), cen[1843:2253], np.zeros(1843)])
    
    # Back project to simplified boundary
    back = np.fft.ifft(np.fft.ifftshift(cen2))
    
    xx = np.round(back.real)
    yy = np.round(back.imag)
    
    m = bw_img.shape[0]
    n = bw_img.shape[1]
    
    xx = xx.astype(np.int)
    yy = yy.astype(np.int)
    
    np.place(xx, xx >= m, m-1)
    np.place(yy, yy >= n, n-1)
    
    simp = np.zeros([m,n])
    simp[xx,yy] = 1
    
    # Fill the simplified boundary
    fill = ndimage.binary_fill_holes(simp).astype(int)
    masked = fill * np.pad(gray, 1, 'constant')

    # Texture descripters
    prob = np.histogram(masked, 256) # assume gray scale with 256 levels
    prob = np.asarray(prob[0]).astype(np.float64)
    prob[0] = 0 # don't count black pixels
    prob = prob / prob.sum()
    vec = np.arange(0, len(prob)).astype(np.float64) / (len(prob) - 1)
    ind = np.nonzero(prob)[0]
    
    # mean grey value
    mu = np.sum(vec[ind] * prob[ind])
    
    # variance 
    var = np.sum((((vec[ind] - mu)**2) * prob[ind]))
    
    # standard deviation
    std =  np.sqrt(var)
    
    # contrast
    cont = 1 - 1/(1 + var)
    
    # 3rd moment
    thir = np.sum(((vec[ind] - mu)**3)*prob[ind])
    
    # Uniformity
    uni = np.sum(prob[ind]**2)
    
    # Entropy
    ent = - np.sum(prob[ind] * np.log2(prob[ind]))
    
    # Add to Feature Vector
    X[18] = mu
    X[19] = std
    X[20] = cont
    X[21] = thir
    X[22] = uni
    X[23] = ent    
                
    # Gray Level Coocurrence Matrix
    # Gray level coocurrence matrix 
    dist = [1,2,4,16,32,64]
    ang = [0, pi/4, pi/2, 3*pi / 4]
    P = greycomatrix(masked, distances = dist, angles = ang, levels=256, normed = True)
    grey_mat = np.zeros([24,2]) 
    flag = 0
    grey_props = ['contrast', 'homogeneity', 'energy', 'correlation']
    for name in grey_props:
        stat = greycoprops(P, name)
        grey_mat[flag:flag+6,0] = np.mean(stat,1)
        grey_mat[flag:flag+6,1] = np.std(stat,1)
        flag += 6
        
    # Add to feature Vector
    X[24:48] = grey_mat[:,0]
    X[48:72] = grey_mat[:,1]
        
    # Morphological Features
    
    # Compute morphological descriptors
    label_img = measure.label(bw_img,neighbors=8,background=0)
    features = measure.regionprops(label_img+1)

    maxArea = 0
    maxAreaInd = 0
    for f in range(0,len(features)):
        if features[f].area > maxArea:
            maxArea = features[f].area
            maxAreaInd = f


    # Compute translation, scale, and rotation invariant features
    ii = maxAreaInd
    aspect = features[ii].minor_axis_length/features[ii].major_axis_length
    area1 = features[ii].area.astype(np.float64)/features[ii].convex_area.astype(np.float64)
    area2 = features[ii].extent
    area3 = features[ii].area.astype(np.float64)/(features[ii].perimeter*features[ii].perimeter)
    area4 = area2/area1
    area5 = area3/area1
    fillArea = features[ii].filled_area
    ecc = features[ii].eccentricity
    esd = features[ii].equivalent_diameter
    en = features[ii].euler_number
    sol = features[ii].solidity
    
    # Add to feature vector
    X[0] = aspect
    X[1] = area1
    X[2] = area2
    X[3] = area3
    X[4] = area4
    X[5] = area5
    X[6] = fillArea
    X[7] = ecc
    X[8] = esd
    X[9] = en
    X[10] = sol
    
    # Hu Moments
    X[11:18] = features[ii].moments_hu
    
    return X

def main():
    imgname = '/data4/plankton_wi17/mpl/source_domain/ensembleClassifier/resized_img.png'
    img = caffe.io.load_image(imgname)
    img = caffe.io.resize_image(img, [256,256])
    img = (img*255).astype(np.uint8)
    featureVector = extractFeatures(img)
    print featureVector.size
    
if __name__ == '__main__':
    main()
