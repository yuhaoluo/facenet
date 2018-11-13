# MIT License
# 
#!/usr/bin/python
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import align.detect_face
import time
import imageio
import cv2
#import skimage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

def load_mtcnn_model(args):
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            start_time = time.time();
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            print('create and load mtcnn model time: ', (time.time() - start_time))
    
    return pnet,rnet,onet
    
def crop_face(img,bounding_boxes,margin,image_size):
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces>0:
        det = bounding_boxes[:,0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces>1:
            for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
        else:
            det_arr.append(np.squeeze(det))
            
        face_res = []
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            #scaled = skimage.transform.resize(cropped, (args.image_size, args.image_size), interp='bilinear')
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            face_res.append(scaled)
    return face_res

def save_faces(res,output_filename):
    if(len(res)>0):
        filename_base, file_extension = os.path.splitext(output_filename)
    for i in range(len(res)):
        if (len(res)>1):
            output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
        else:
            output_filename_n = "{}{}".format(filename_base, file_extension)
        imageio.imwrite(output_filename_n, res[i])

def img_resize(img,scale):
    
    img_resize = misc.imresize(img, (int(img.shape[0]/scale), int(img.shape[1]/scale)), interp='bilinear')
    return img_resize

def faceDetect(args,img,pnet,rnet,onet,isPrintTimeInfo=False):
    ## reseize img to detect
    if args.scale>1:
        img_input = img_resize(img,args.scale)
    else:
        img_input = img
    
    detect_time_start = time.time()
    bounding_boxes, _ = align.detect_face.detect_face(img_input, minsize, pnet, rnet, onet, threshold, factor)
    if args.scale>1:
        bounding_boxes[:,0:4] = args.scale * bounding_boxes[:,0:4]
        
    detect_time = time.time() - detect_time_start
    if isPrintTimeInfo:
        print('detect_face_time: ', detect_time)  
        
    #print(bounding_boxes)
    return bounding_boxes

    



########################################################################################################
def main(args):
      
    pnet,rnet,onet = load_mtcnn_model(args)
    # Create a video capture object to read videos
    cap = cv2.VideoCapture(args.video_path)
    while True:
        ok,frame = cap.read()
        
        if not ok:
            break
        timer = cv2.getTickCount()
        boxes = faceDetect(args,frame,pnet,rnet,onet)
        
        fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
        
        #print(fps)
        
        if boxes.shape[0]>0:
            bb = np.squeeze(boxes[:,0:4])
            if bb.size == 4 :
                p1 = (int(bb[0]),int(bb[1]))
                p2 = (int(bb[2]),int(bb[3]))
                cv2.rectangle(frame,p1,p2,(255,0,0),2,1)
                cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
                # show frame
                cv2.imshow('MultiTracker', frame)
                    # quit on ESC button
                if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                    break
        #draw = align.detect_face.drawBoxes(frame,boxes)
        #imageio.imwrite("/home/luoyuhao/Datasets/Tracking/video/res.png",draw)
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('video_path', type=str, help='Directory with unaligned images.')
    #parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    parser.add_argument('--scale', type=int,
                        help='the height and width will resize to height/scale and width/scale to detect faces.', default=2)
    return parser.parse_args(argv)

if __name__ == '__main__':

    img_path2 = "/home/luoyuhao/Datasets/Tracking/video/res.png"
    output_dir2 = "/home/luoyuhao/Datasets/Tracking/video/res"
    
    video_path = "/home/luoyuhao/Datasets/Tracking/video/4.mp4"
    args = [video_path,'--scale','2']
   
    main(parse_arguments(args))
    #main(parse_arguments(sys.argv[1:]))
    
# =============================================================================
#     img =  cv2.imread(img_path2)
#     cv2.imshow("",img)
# 
#     cv2.waitKey(0) 
#     cv2.destroyAllWindows()
# =============================================================================
   
