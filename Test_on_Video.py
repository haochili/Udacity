from moviepy.editor import VideoFileClip
#from IPython.display import HTML
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
from scipy.ndimage.measurements import label
from collections import deque

dist_pickle = pickle.load(open("svc_search_classify_only.p", "rb" ))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

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
y_start_stop96 = [400, 600] # Min and max in y to search in slide_window()
y_start_stop64 = [400, 600]
y_start_stop32 = [400, 500]
#iteration=1
heatmaps = deque(maxlen=10)

def process_video(image):
    #print('Start to test with single image')
    #from lesson_functions_hog import *
    t=time.time()
    # Signal image testing  
    draw_image = np.copy(image)/255

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    draw_image = draw_image.astype(np.float32)
    t2=time.time()
    #print(round(t2-t, 2), 'Seconds to load a singal image')
    windows = []
    win96 = slide_window(draw_image, x_start_stop=[None, None], y_start_stop=y_start_stop96, 
                    xy_window=(128, 128), xy_overlap=(0.7, 0.7))
    win64 = slide_window(draw_image, x_start_stop=[None, None], y_start_stop=y_start_stop64, 
                    xy_window=(64, 64), xy_overlap=(0.8, 0.8))
    win32 = slide_window(draw_image, x_start_stop=[None, None], y_start_stop=y_start_stop32, 
                    xy_window=(32, 32), xy_overlap=(0.7, 0.7))
    #find all possible windows and return a list of start/end coordinates
    windows.extend(win96)
    windows.extend(win64)
    windows.extend(win32)
    cimg = np.copy(image)
    #windows = search_with_multiscale_windows(image, cspace, orient, pix_per_cell, cell_per_block,point_scale_data)
    hot_windows = search_windows(draw_image, windows, svc, X_scaler, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    #print("iterations: ",iteration)
    global heatmaps #,iteration
    if len(hot_windows) > 0:
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = add_heat(heat, hot_windows)
        heatmaps.append(heat)
        #print(len(heatmaps))
        # take recent 10 heatmaps and average them
        if len(heatmaps) == 10:
            avg_heat = sum(heatmaps)/len(heatmaps)
            heat = avg_heat
        heat = apply_threshold(heat,8)
        heatmap = np.clip(heat, 0, 255)
        #heatmaps.append(heatmap)
        #if iteration % 10 == 0:
        #    avg_heatmaps = sum(heatmaps)/len(heatmaps)
        #    heatmap = avg_heatmaps
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(cimg, labels)
    else:
        # pass the image itself if nothing was detected
        draw_img = cmig
    #iteration += 1
    return draw_img

# keep current heatmaps
white_output = 'result.mp4'
#clip1 = VideoFileClip("test_video.mp4").subclip(0,1)
#clip1 = VideoFileClip("test_video.mp4")
clip1 = VideoFileClip("project_video.mp4")
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_video) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
