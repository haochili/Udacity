import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions_search_classifier import *
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

# Read in cars and notcars
images_path = glob.glob('./*vehicle*/*/*')
print(len(images_path))
#print(images_path[0:len(images_path)])
print(images_path[-1])
cars = []
notcars = []
for image in images_path:
    if 'non-vehicles' in image:
        notcars.append(image)
    else:
        cars.append(image)

np.random.shuffle(cars)
np.random.shuffle(notcars)
print('cars', len(cars))
print('non-cars', len(notcars))

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
#sample_size = 1000
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (64, 64) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [300, 700] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
print('Shape of training data X is:',X.shape)
# Fit a per-column scaler, Haochi: X_scaler is used to store X's mean and std value, but doesn't change X until transform(X) takes place
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X, Haochi: use the mean and std value stored in X_scaler to transform X into the normalized scaled_X data set
scaled_X = X_scaler.transform(X)
print('Shape of the scaled training data scaled_X is:',scaled_X.shape)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
print('Shape of the feature label y is:',y.shape)
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
print('rand_state is:',rand_state)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('length of X_train:', len(X_train))
print('Feature vector length:', len(X_train[0]))
print('length of X_test:', len(X_test))
print('Feature vector length:', len(X_test[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
filename = 'svc_search_classify_only.p'
#filename = 'svc_pickle_small_sample.p'
#pickle.dump(scale, open(filename, 'wb'))
dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins
pickle.dump(dist_pickle, open(filename, 'wb'))

#filename.close()
print('Model saved')
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train save model to pickle')
print()
'''
print('Start to test with single image')
from lesson_functions_hog import *
t=time.time()
# Signal image testing  
image = mpimg.imread('test1.jpg')
draw_image = np.copy(image)/255

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
draw_image = draw_image.astype(np.float32)
t2=time.time()
print(round(t2-t, 2), 'Seconds to load a singal image')
windows = slide_window(draw_image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))
#find all possible windows and return a list of start/end coordinates

hot_windows = search_windows(draw_image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                   

#Search window returns a list of window coordinates- part of "windows" list that is is determined to have cars features in the content of the window area. Search_window function makes feature vector for each window covered image, does feature rescale- to the same sacle of traning image, fit into classifer, and make preditions of the result. 

window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)
print('image shape:',np.shape(window_img))
print(np.max(window_img))
heat = np.zeros_like(image[:,:,0])
heat = add_heat(heat,hot_windows)
heat = apply_threshold(heat,1)
heatmap = np.clip(heat, 0, 255)
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)
t3 = time.time()
print(round(t3-t2, 2), 'Seconds to find cars in the singal image')
fig = plt.figure()
plt.subplot(221)
aa=plt.imshow(window_img)
plt.title('Test Image Outputs')
plt.subplot(222)
ab=plt.imshow(draw_img)
plt.title('Cut False Positive')
plt.subplot(223)
cd=plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
#plt.show(ab)
plt.show(fig)
'''

