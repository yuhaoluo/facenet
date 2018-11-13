# MIT License
#
# Copyright (c) 2017 PXL University College
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

# Clusters similar faces from input folder together in folders based on euclidean distance matrix

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import time
import os
import sys
import argparse
import facenet
import align.detect_face
from sklearn.cluster import DBSCAN


def std_img(img_list):
    img_list_out = []
    for i in range(len(img_list)):
        prewhitened = facenet.prewhiten(img_list[i])
        img_list_out.append(prewhitened)
    if len(img_list_out) > 0:
        images = np.stack(img_list_out)
        return images
    else:
        return None

def main(args):
    pnet, rnet, onet = create_network_face_detection(args.gpu_memory_fraction)

    image_list = load_images_from_folder(args.data_dir) #imgage_list, type:list
    images = align_data(image_list, args.image_size, args.margin, pnet, rnet, onet)



def align_data(image_list, image_size, margin, pnet, rnet, onet):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []
    total_time = 0
    for x in range(len(image_list)):
        img_size = np.asarray(image_list[x].shape)[0:2]
        start_time = time.time()
        bounding_boxes, _ = align.detect_face.detect_face(image_list[x], minsize, pnet, rnet, onet, threshold, factor)
        total_time += time.time() - start_time
        
    print("total_detection_time: %.2f"%total_time)
    print("average_detection_time: %.2f"% (total_time/len(image_list)))
# =============================================================================
#         np.save("./bbs/crop/img"+str(x),bounding_boxes[:,0:4])
#         nrof_samples = len(bounding_boxes)
#         if nrof_samples > 0:
#             for i in range(nrof_samples):
#                 if bounding_boxes[i][4] > 0.95:
#                     det = np.squeeze(bounding_boxes[i, 0:4])
#                     bb = np.zeros(4, dtype=np.int32)
#                     bb[0] = np.maximum(det[0] - margin / 2, 0)
#                     bb[1] = np.maximum(det[1] - margin / 2, 0)
#                     bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
#                     bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
#                     cropped = image_list[x][bb[1]:bb[3], bb[0]:bb[2], :]
#                     aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
#                     
#                     if not os.path.exists('./bbs/crop'):
#                         os.mkdir('./bbs/crop')
#                     misc.imsave('./bbs/crop/'+str(i)+'.png',aligned)
#                     #prewhitened = facenet.prewhiten(aligned)
#                     img_list.append(aligned)
# =============================================================================

    if len(img_list) > 0:
        images = np.stack(img_list)
        return images
    else:
        return None


def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = misc.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='The directory containing the images to cluster into folders.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--min_cluster_size', type=int,
                        help='The minimum amount of pictures required for a cluster.', default=1)
    parser.add_argument('--cluster_threshold', type=float,
                        help='The minimum distance for faces to be in the same cluster', default=1.0)
    parser.add_argument('--largest_cluster_only', action='store_true',
                        help='This argument will make that only the biggest cluster is saved.')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
