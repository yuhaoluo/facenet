# MIT License
# 
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
import face_preprocess
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
    face_res = []
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
    
def crop_face_with_align(img,bounding_boxes,landmark,margin,image_size):
    nrof_faces = bounding_boxes.shape[0]
    face_res = []
    if nrof_faces>0:
        det = bounding_boxes[:,0:4]
        det_arr = []
        if nrof_faces>1:
            for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
        else:
            det_arr.append(np.squeeze(det))
            
        for i, det in enumerate(det_arr):
            point = landmark[:,i]
            
            point_x = point[0:5]
            point_y = point[5:10]
            mark = np.zeros((5,2))
            mark[:,0] = point_x
            mark[:,1] = point_y
            #print(arr2)      
            cropped = face_preprocess.preprocess(img, det, mark, image_size='112,112')                        
            face_res.append(cropped)
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

def detectFace(args,img,output_filename=None,isDrawFace=False,isPrintTimeInfo=False):
    ## reseize img to detect
    if args.scale>1:
        img_input = img_resize(img,args.scale)
    else:
        img_input = img
    
    pnet,rnet,onet = load_mtcnn_model(args)

    detect_time_start = time.time()
    
    bounding_boxes, points = align.detect_face.detect_face(img_input, minsize, pnet, rnet, onet, threshold, factor)
    if args.scale>1:
        bounding_boxes[:,0:4] = args.scale * bounding_boxes[:,0:4]
        points = args.scale * points
        
    detect_time = time.time() - detect_time_start
    if isPrintTimeInfo:
        print('detect_face_time: ', detect_time)  
        
    faces = crop_face_with_align(img,bounding_boxes,points,args.margin,args.image_size)
    #faces = crop_face(img,bounding_boxes,args.margin,args.image_size)
# =============================================================================
#     if(output_filename is not None):
#         save_time_start = time.time()
#         save_faces(faces,output_filename)
#         if isPrintTimeInfo:
#             print('save_face_time: ', time.time() - save_time_start) 
# =============================================================================
    if isDrawFace:
        draw = align.detect_face.drawBoxes(img,bounding_boxes)
        filename_base, file_extension = os.path.splitext(output_filename)
        imageio.imwrite(filename_base+'_res'+file_extension,draw)
    #print(bounding_boxes)
    return faces,bounding_boxes

    



########################################################################################################
def main(args):
# =============================================================================
#     try:
#         img = misc.imread(image_path)
#     except(IOError,ValueError, IndexError) as e:
#         errorMessage = '{}: {}'.format(image_path, e)
#         print(errorMessage)
#     else:
#         if img.ndim<2:
#             print('Unable to align "%s"' % image_path)
#             #continue         
#         if img.ndim == 2:
#             img = facenet.to_rgb(img)
#         img = img[:,:,0:3]
# =============================================================================
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)          
    #
    image_path = args.image_path
    img = imageio.imread(image_path)
    filename = os.path.splitext(os.path.split(image_path)[1])[0]
    output_filename = os.path.join(output_dir, filename+'.png')
    
    faces,_ = detectFace(args,img,output_filename,True,True)
    
    for i in range(len(faces)):
        imageio.imsave(output_dir+'/_'+str(i)+'.png',faces[i])
# =============================================================================
#     img_input = []
#     if args.scale>1:
#         img_input = img_resize(img,args.scale)
#     else:
#         img_input = img
# 
#     filename = os.path.splitext(os.path.split(image_path)[1])[0]
#     output_filename = os.path.join(output_dir, filename+'.png')
# 
#     pnet,rnet,onet = load_mtcnn_model(args)
# 
# 
#     detect_time_start = time.time()
#     bounding_boxes, _ = align.detect_face.detect_face(img_input, minsize, pnet, rnet, onet, threshold, factor)
#     if args.scale>1:
#         bounding_boxes[:,0:4] = args.scale * bounding_boxes[:,0:4]
#     detect_time = time.time() - detect_time_start
#     print('detect_face_time: ', detect_time)  
#     
#     
#     res = crop_face(img,bounding_boxes,args.margin,args.image_size)
#     
# # =============================================================================
# #     save_time_start = time.time()
# #     save_faces(res,output_filename)
# #     print('save_face_time: ', time.time() - save_time_start) 
# # =============================================================================
#     
#     draw = align.detect_face.drawBoxes(img,bounding_boxes)
#     filename_base, file_extension = os.path.splitext(output_filename)
#     imageio.imwrite(filename_base+'_res'+file_extension,draw)
#     
#     #text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
# =============================================================================

##########################################################################################################                            

   

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
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
    
    img_path2 = '/home/luoyuhao/Documents/Code/Test/mxnet_facenet/20181108/align_pic/22-1541583947376-347.jpg'
    output_dir2 = '/home/luoyuhao/Documents/Code/Test/mxnet_facenet/20181108/align_pic/res'

    img_path = '/home/luoyuhao/Datasets/ai_cloud_evaluate/src/d3.jpg'
    output_dir = '/home/luoyuhao/Datasets/ai_cloud_evaluate/src'
    args = [img_path,output_dir,'--scale','1']
    main(parse_arguments(args))
    #main(parse_arguments(sys.argv[1:]))
   
