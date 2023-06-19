#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#import tensorflow as tf
import sys
import time
import struct
import threading

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import h5py
import math
from termcolor import colored, cprint  
import copy
import tensorflow as tf
#from shapely.geometry import MultiPoint
import torch
import torch.nn.functional as F

#-----------------------------------------------------------------------------
# data argumentation
#-----------------------------------------------------------------------------
def perturb(input_image, pixels, set_theta_zero=False):
    """Data augmentation on images."""
    image_size = input_image.shape[:2]

    # Compute random rigid transform.
    while True:
        theta, trans, pivot = get_random_image_transform_params(image_size)
        if set_theta_zero:
            theta = 0.
        transform = get_image_transform(theta, trans, pivot)
        transform_params = theta, trans, pivot

        # Ensure pixels remain in the image after transform.
        is_valid = True
        new_pixels = []
        for pixel in pixels:
            pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)
            pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
            pixel = np.flip(pixel)
            in_fov = pixel[0] < image_size[0] and pixel[1] < image_size[1]
            is_valid = is_valid and np.all(pixel >= 0) and in_fov
            new_pixels.append(pixel)
        if is_valid:
            break

    # Apply rigid transform to image and pixel labels.
    input_image = cv.warpAffine(input_image, transform[:2, :],
                                 (image_size[1], image_size[0]),
                                 flags=cv.INTER_NEAREST)
    return input_image, new_pixels


def get_random_image_transform_params(image_size):
    theta_sigma = 2 * np.pi / 6
    theta = np.random.normal(0, theta_sigma)

    trans_sigma = np.min(image_size) / 6
    trans = np.random.normal(0, trans_sigma, size=2)  # [x, y]
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot

def get_image_transform(theta, trans, pivot=[0, 0]):
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_T_image = np.array([[1., 0., -pivot[0]],
                              [0., 1., -pivot[1]],
                              [0., 0.,        1.]])
    image_T_pivot = np.array([[1., 0., pivot[0]],
                              [0., 1., pivot[1]],
                              [0., 0.,       1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]],
                          [0.,            0.,            1.]])
    return np.dot(image_T_pivot, np.dot(transform, pivot_T_image))


#-----------------------------------------------------------------------------
# 3D heatmap
#-----------------------------------------------------------------------------

def reconstruct_heightmaps(color, depth, configs, bounds, pixel_size):
    heightmaps, colormaps = [], []
    for color, depth, config in zip(color, depth, configs):
        intrinsics = np.array(config['intrinsics']).reshape(3, 3)
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config['position']).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
        heightmaps.append(heightmap)
        colormaps.append(colormap)
    return heightmaps, colormaps

def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in camera coordinates.
      transform: 4x4 float array representing a rigid transformation matrix.

    Returns:
      points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            'constant', constant_values=1)
    for i in range(3):
        points[..., i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points

def get_heightmap(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in world coordinates.
      colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      pixel_size: float defining size of each pixel in meters.

    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[..., 0] >= bounds[0, 0]) & (points[..., 0] < bounds[0, 1])
    iy = (points[..., 1] >= bounds[1, 0]) & (points[..., 1] < bounds[1, 1])
    iz = (points[..., 2] >= bounds[2, 0]) & (points[..., 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
    return heightmap, colormap

def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.

    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points

#-----------------------------------------------------------------------------
# pixel utils
#-----------------------------------------------------------------------------

def pixel_to_position(pixel, height, camera_config,  pixel_size):
    """Convert from pixel  to world."""
    camera_pos = camera_config['position']
    image_width = camera_config['image_size'][1]
    image_height = camera_config['image_size'][0]
    u =  pixel[0]
    v =  pixel[1]
    x = camera_pos[0] + (v-image_height/2) * pixel_size
    y = camera_pos[1] + (u-image_width/2) * pixel_size
    z = camera_pos[2] - height
    if z < 0:
        z = 0
    return [x, y, z]


def position_to_pixel(position, camera_config, pixel_size):
    """Convert from 3D position to pixel location on heightmap."""
    camera_pos = camera_config['position']
    image_width = camera_config['image_size'][1]
    image_height = camera_config['image_size'][0]
    u = int(image_width/2 + (position[1] - camera_pos[1]) / pixel_size)
    v = int(image_height/2 + (position[0] - camera_pos[0]) / pixel_size)
    if u >= image_width:
        u = image_width-1
    if v >= image_height:
        v = image_height-1
    return [u, v]

def position_to_bound_pixel(position, bounds, pixel_size):
    """Convert from 3D position to pixel location on heightmap."""
    u = int(np.round((position[1] - bounds[1, 0]) / pixel_size))
    v = int(np.round((position[0] - bounds[0, 0]) / pixel_size))
    return (u, v)
    

def bound_pixel_to_position(pixel, height, bounds, pixel_size, skip_height=False):
    """Convert from pixel location on heightmap to 3D position."""
    u, v = pixel #u,v
    x = bounds[0, 0] + v * pixel_size
    y = bounds[1, 0] + u * pixel_size
    if not skip_height:
        z = bounds[2, 0] + height[u, v]
    else:
        z = 0.0
    return (x, y, z)

#-----------------------------------------------------------------------------
# IMAGE UTILS
#-----------------------------------------------------------------------------

def preprocess_color(image, mean=0.5, std=0.225):
    image = (image.copy() / 255 - mean) / std
    return image


def preprocess_depth(image, mean=0.005, std=0.008):
    image = (image.copy() - mean) / std
    image = np.tile(image.reshape(
        image.shape[0], image.shape[1], 1), (1, 1, 3))
    return image


#-----------------------------------------------------------------------------
# PLOT UTILS
#-----------------------------------------------------------------------------

# Plot colors (Tableau palette).
COLORS = {'blue': [078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0],
                   'red':  [255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0],
                   'green':  [089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0],
                  'orange': [242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0],
                 'yellow': [237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0],
                 'purple': [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0],
                 'pink':   [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0],
                 'cyan':   [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0],
                'brown':  [156.0 / 255.0, 117.0 / 255.0, 095.0 / 255.0],
                'gray':   [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0]}


#-----------------------------------------------------------------------------
# geometry
#-----------------------------------------------------------------------------

ring_templates = {
                   'ring': [[319,308],[333,306],[345,302],[358,297],[367,287],[375,277],[382,265],[386,253],
                              [388,239],[387,226],[383,213],[376,201],[368,191],[357,183],[345,176],[333,172],
                              [319,171],[306,172],[293,176],[281,183],[271,191],[263,202],[256,213],[252,226],
                              [251,240],[252,253],[256,266],[262,278],[271,288],[281,297],[293,303],[306,307]],
                }

rope_templates = {
                   'line': [[196,239],[208,239],[221,239],[234,239],
                            [247,239],[260,239],[273,239],[286,239],
                            [299,239],[312,239],[325,239],[337,239],
                            [350,239],[363,239],[376,239],[389,239]],
                    'l_shape':[[196,239],[208,239],[221,239],[234,239],
                            [247,239],[260,239],[273,239],[286,239],
                            [299,239],[312,239],[325,239],[337,239],
                            [337,252],[337,265],[337,278],[337,291]],
                             }

cloth_templates = {'square':[[-2,2],[-1,2],[0,2],[1,2],[2,2], 
                            [-2,1],[-1,1],[0,1],[1,1],[2,1],
                            [-2,0],[-1,0],[0,0],[1,0],[2,0],
                            [-2,-1],[-1,-1],[0,-1],[1,-1],[2,-1],
                            [-2,-2],[-1,-2],[0,-2],[1,-2],[2,-2]],
                  'triangle':[[-2,2],[-2,1],[-2,0],[-2,-1],[-2,-2], 
                              [-2,1],[-1,1],[-1,0],[-1,-1],[-1,-2],
                              [-2,0],[-1,0],[0,0],[0,-1],[0,-2],
                              [-2,-1],[-1,-1],[0,-1],[1,-1],[1,-2],
                              [-2,-2],[-1,-2],[0,-2],[1,-2],[2,-2]],
                  'rhombus':[[0,0],[0,1],[0,2],[0,1],[0,0], 
                            [-1,0],[-1,1],[0,1],[1,1],[1,0],
                            [-2,0],[-1,0],[0,0],[1,0],[2,0],
                            [-1,0],[-1,-1],[0,-1],[1,-1],[1,0],
                            [0,0],[0,-1],[0,-2],[0,-1],[0,0]],
                  'half':[[-2,2],[-1,2],[0,2],[1,2],[2,2], 
                          [-2,1],[-1,1],[0,1],[1,1],[2,1],
                          [-2,0],[-1,0],[0,0],[1,0],[2,0],
                          [-2,1],[-1,1],[0,1],[1,1],[2,1],
                          [-2,2],[-1,2],[0,2],[1,2],[2,2]],
                 'quarter':[[-2,2],[-1,2],[0,2],[-1,2],[-2,2], 
                            [-2,1],[-1,1],[0,1],[-1,1],[-2,1],
                            [-2,0],[-1,0],[0,0],[-1,0],[-2,0],
                            [-2,1],[-1,1],[0,1],[-1,1],[-2,1],
                            [-2,2],[-1,2],[0,2],[-1,2],[-2,2]]
                        }

       

def kb(vertex1, vertex2):
    x1 = vertex1[0]
    y1 = vertex1[1]
    x2 = vertex2[0]
    y2 = vertex2[1]  
    if x1==x2:
        return (0, x1)      
    if y1==y2:              
        return (1, y1)      
    else:
        k = (y1-y2)/(x1-x2)
        b = y1 - k*x1
        return (2, k, b)    


def isConvex(vertexes):
    convex = True    
    l = len(vertexes)
    if l<3:
        raise ValueError()
    
    for i in range(l):
        pre = i
        nex = (i+1)%l
        
        line = kb(vertexes[pre], vertexes[nex])
        
        if line[0]==0:
            offset = [vertex[0]-vertexes[pre][0] for vertex in vertexes]
        elif line[0]==1:
            offset = [vertex[1]-vertexes[pre][1] for vertex in vertexes]
        else:
            k, b = line[1], line[2]
            offset = [k*vertex[0]+b-vertex[1] for vertex in vertexes]
        
        for o in offset:
            for s in offset:
                if o*s<0:
                    convex = False
                    break
            if convex==False:
                break
                    
        if convex==False:
            break

    return convex

def line_cross(p1,p2,p3):
    x1=p2[0]-p1[0]
    y1=p2[1]-p1[1]
    x2=p3[0]-p1[0]
    y2=p3[1]-p1[1]
    return x1*y2-x2*y1 

def IsIntersec(p1,p2,p3,p4): 

    if(max(p1[0],p2[0])>=min(p3[0],p4[0])    
    and max(p3[0],p4[0])>=min(p1[0],p2[0])   
    and max(p1[1],p2[1])>=min(p3[1],p4[1])   
    and max(p3[1],p4[1])>=min(p1[1],p2[1])): 

        if(line_cross(p1,p2,p3)*line_cross(p1,p2,p4)<=0
           and line_cross(p3,p4,p1)*line_cross(p3,p4,p2)<=0):
            D=1
        else:
            D=0
    else:
        D=0
    return D

def isCross(vertexes):
    convex = True    
    l = len(vertexes)
    for i in range(l-3):
        for j in range(i+2,l-1,1):
            if IsIntersec(vertexes[i],vertexes[i+1],vertexes[j],vertexes[j+1]):
                convex = False
                break
    return convex

def isAngle(vertexes):
    angle = True
    for i in range(len(vertexes)-2):
        arr_a = np.array([(vertexes[i+1][0]- vertexes[i][0]), (vertexes[i+1][1]- vertexes[i][1])])  
        arr_b = np.array([(vertexes[i+1][0]- vertexes[i+2][0]), (vertexes[i+1][1]- vertexes[i+2][1])])
        if (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))) == 0:
            angle = False
            break
        cos_value = (float(arr_a.dot(arr_b)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))))  # 注意转成浮点数运算
        if cos_value >1:
            cos_value = 1
        elif cos_value <-1:
            cos_value = -1
        cross_angle = np.arccos(cos_value) * (180 / np.pi)
        if  cross_angle<30:
            angle = False
            break
    return angle

def isAngle_r(vertexes):
    angle = True
    point_num = len(vertexes)
    for i in range(len(vertexes)):
        arr_a = np.array([(vertexes[(i+1)%point_num][0]- vertexes[i][0]), (vertexes[(i+1)%point_num][1]- vertexes[i][1])])  
        arr_b = np.array([(vertexes[(i+1)%point_num][0]- vertexes[(i+2)%point_num][0]), (vertexes[(i+1)%point_num][1]- vertexes[(i+2)%point_num][1])])
        if (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))) == 0:
            angle = False
            break
        cos_value = (float(arr_a.dot(arr_b)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))))  
        if cos_value >1:
            cos_value = 1
        elif cos_value <-1:
            cos_value = -1
        cross_angle = np.arccos(cos_value) * (180 / np.pi)
        if  cross_angle<40:
            angle = False
            break
    return angle

def load_data_from_h5(h5_file_name,model_type='feature_extract'):
    h5_file = h5py.File(h5_file_name,'r')

    if model_type == 'feature_extract':
        c_image = []
        n_image = []
        for a_c in h5_file["c_image"]:
            c_image.append(a_c)
        for n_c in h5_file["n_image"]:
            n_image.append(n_c)
        c_np = np.array(c_image)/255.0
        n_np = np.array(n_image)/255.0
        c_np = c_np.astype(np.float32)
        n_np = n_np.astype(np.float32)
        h5_file.close()
        return np.concatenate((c_np[:,np.newaxis,:,:,:], n_np[:,np.newaxis,:,:,:]), axis=1),n_np

    elif model_type == 'transporter_graph':
        c_image =  h5_file["c_image"]
        g_image =  h5_file["g_image"]         
        c_np = np.array(c_image)/255.0
        g_np = np.array(g_image)/255.0
        c_np = c_np.astype(np.float32)
        g_np = g_np.astype(np.float32)
        pick = h5_file['pick']
        place = h5_file['place']
        return np.concatenate((c_np,g_np),axis=3), pick, place,c_np,g_np


#-----------------------------------------------------------------------------
# keypoint extracting model
#-----------------------------------------------------------------------------

def show_keypoint(image,keypoint_list):
    keypoint_x_list = []
    keypoint_y_list = []
    for keypoint in keypoint_list:
        x = keypoint[0]*120 + 120
        y = keypoint[1]*160 + 160
        if x == 240:
            x=239
        if y == 320:
            y=319
        keypoint_x_list.append(x)
        for i in range(0,4):
            for j in range(0,4):
                draw_x = int(x-4 + i)
                draw_y = int(y-4 + j)
                image[draw_x][draw_y] = [240,65,85]
    return image
    #implot = plt.imshow(image)
    #plt.scatter(keypoint_y_list, keypoint_x_list, s=30, c='r')
    #plt.axis('off')
    #plt.plot(keypoint_y_list,keypoint_x_list,'o',s=20, c='b')
    #plt.show()

def get_gaussian_maps(mu, map_size, inv_std=15): #15 2d 20 1d
    """
      Args:
      mu: A tensor of shape [B, K, 2] of the y, x center points of each keypoint
      map_size: gaussian_map size [H,W]

      Returns:
      gaussian_maps  A tensor of shape [B, H, W, K]
    """

    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    mu_y = mu_y.unsqueeze(-1)
    mu_x = mu_x.unsqueeze(-1)

    y = torch.linspace(-1.0,1.0,steps=map_size[0]).cuda()
    x = torch.linspace(-1.0,1.0,steps=map_size[1]).cuda()

    y = y.view(1, 1, map_size[0], 1)
    x = x.view(1, 1, 1, map_size[1])

    g_y = torch.pow(y - mu_y, 2)
    g_x = torch.pow(x - mu_x, 2)
    dist = (g_y + g_x) * inv_std *inv_std
    g_yx = torch.exp(-dist).float()
  
    g_yx = g_yx.permute(0, 2, 3, 1)
    g_yx = torch.sum(g_yx,dim=3)
    return g_yx

def _get_gaussian_maps(mu, map_size, inv_std, power=2):
  """Transforms the keypoint center points to a gaussian masks."""
  mu_y, mu_x = 2*(mu[:, :, 1:2]-0.5*map_size[0])/map_size[0], 2*(mu[:, :, 0:1]-0.5*map_size[1])/map_size[1]
  y = tf.cast(tf.linspace(-1.0, 1.0, map_size[0]), tf.float32)
  x = tf.cast(tf.linspace(-1.0, 1.0, map_size[1]), tf.float32)

  mu_y, mu_x = tf.expand_dims(mu_y, -1), tf.expand_dims(mu_x, -1)

  y = tf.reshape(y, [1, 1, map_size[0], 1])
  x = tf.reshape(x, [1, 1, 1, map_size[1]])

  g_y = tf.pow(y - mu_y, power)
  g_x = tf.pow(x - mu_x, power)
  dist = (g_y + g_x) * tf.pow(inv_std, power)
  g_yx = tf.exp(-dist)
  g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])
  return g_yx



def get_coord(features):
    """
      Args:
        features: A tensor of shape [B, K, F_h, F_w] where K is the number of keypoints to extract.

      Returns:
        A tensor of shape [B,K,2] containing the keypoint centers. The location is given in the range [-1, 1].
    """
    features = features.permute(0, 2, 3, 1)
    y_axis_size = features.shape[1]
    x_axis_size = features.shape[2]

    # Compute the normalized weight for each row/column along the axis
    g_c_prob_y = features.mean(dim=2)
    g_c_prob_y =F.softmax(g_c_prob_y, dim=1)
    scale_y = torch.linspace(-1.0,1.0,steps=y_axis_size)
    scale_y = scale_y.view(1, y_axis_size, 1).cuda()
    coordinate_y = torch.sum(g_c_prob_y*scale_y,dim=1)

    g_c_prob_x = features.mean(dim=1)
    g_c_prob_x =F.softmax(g_c_prob_x, dim=1)
    scale_x = torch.linspace(-1.0,1.0,steps=x_axis_size)
    scale_x = scale_x.view(1, x_axis_size, 1).cuda()
    coordinate_x = torch.sum(g_c_prob_x*scale_x,dim=1)
    coordinate = torch.cat((coordinate_y.unsqueeze(-1),coordinate_x.unsqueeze(-1)),2)

    return coordinate

#-----------------------------------------------------------------------------
# graph-realated
#-----------------------------------------------------------------------------

def get_effect_points(keypoint_list):
    dis_list = []
    keypoints_np = np.array(keypoint_list)
    for keypoint in keypoints_np:
        dis = math.sqrt((keypoint[0])**2+(keypoint[1])**2)
        dis_list.append(dis)
    dis_np = np.array(dis_list)
    effi_idx = np.argpartition(dis_np, 6)
    choose_index = effi_idx[0:6]
    return keypoints_np[choose_index]

def get_two_nearest_point(i,keypoint_list):
    k_1 = keypoint_list[i]
    dis_list = []
    for j in range(len(keypoint_list)) :
        if j!=i:
            k_2 = keypoint_list[j]
            dis = math.sqrt((k_2[0]-k_1[0])**2+(k_2[1]-k_1[1])**2)
        else:
            dis = 10000
        dis_list.append(dis)
    dis_np = np.array(dis_list)
    #print(dis_np)
    effi_idx = np.argpartition(dis_np, 2)
    choose_index = effi_idx[0:2]
    return choose_index[0],choose_index[1]


def get_edge_matrix(keypoint_list):
    point_num = len(keypoint_list)
    matrix = np.zeros((point_num,point_num))
    for i in range(point_num):
        p,k=get_two_nearest_point(i,keypoint_list)
        matrix[i][p] = 1
        matrix[p][i] = 1
        matrix[i][k] = 1
        matrix[k][i] = 1
        matrix[i][i] = 1
    return matrix

def get_edge(action_dim,batch_size):
    edge_index_np = np.zeros((2,action_dim*action_dim*batch_size))
    for b in range(batch_size):
        b_start = b*action_dim*action_dim
        for i in range(action_dim):
            edge_index_np[0][(b_start+action_dim*i):(b_start+action_dim*(i+1))] = b_start/action_dim +i
            for j in range(action_dim):
                edge_index_np[1][b_start+action_dim*i+j] = b_start/action_dim +j
    return edge_index_np

def tosquare(matrix_origin):
    height = matrix_origin.shape[0]
    width =  matrix_origin.shape[1]
    ratio = int(height/width)
    matrix_square = np.zeros((height,height))
    for i in range(ratio):
        matrix_square[i*width:(i+1)*width,i*width:(i+1)*width] = matrix_origin[i*width:(i+1)*width,:]
    return matrix_square


'''    
def get_edge_matrix(keypoint_list):
    point_num = len(keypoint_list)
    matrix = np.zeros((point_num,point_num))
    for i in range(point_num):
        p,k=get_two_nearest_point(i,keypoint_list)
        matrix[i][p] = 1
        matrix[p][i] = 1
        matrix[i][k] = 1
        matrix[k][i] = 1
        matrix[i][i] = 1
    tem_d = np.sum(matrix,axis=1)
    d = np.zeros((point_num,point_num))
    for i in range(len(tem_d)):
        d[i][i]=1.0/np.sqrt(tem_d[i])
    support = np.matmul(np.matmul(d,matrix),d)
    return support
'''


# training process

def print_state(current_points,target_points):
    current_points_x = current_points[:,0]
    current_points_y = current_points[:,1]
    target_points_x = target_points[:,0]
    target_points_y = target_points[:,1]
    x_max_value = max(np.max(current_points_x),np.max(target_points_x))
    x_min_value = min(np.min(current_points_x),np.min(target_points_x))
    y_max_value = max(np.max(current_points_y),np.max(target_points_y))
    y_min_value = min(np.min(current_points_y),np.min(target_points_y))

    current_points_plot = copy.deepcopy(current_points)
    target_points_plot = copy.deepcopy(target_points)
    
    current_points_plot[:,0] = (current_points_plot[:,0]-x_min_value)*19/(x_max_value-x_min_value)
    current_points_plot[:,1] = (current_points_plot[:,1]-y_min_value)*19/(y_max_value-y_min_value)
    current_points_np = current_points_plot.astype('int')

    target_points_plot[:,0] = (target_points_plot[:,0]-x_min_value)*19/(x_max_value-x_min_value)
    target_points_plot[:,1] = (target_points_plot[:,1]-y_min_value)*19/(y_max_value-y_min_value)
    target_points_np = target_points_plot.astype('int')

    color_list = ['white','red','blue','green']  #other:0,current:1,target:2,'cover'
    draw_list = np.zeros((20,20)).astype('int')
    for kp1 in current_points_np:
        draw_list[kp1[0]][kp1[1]] = 1
    for kp2 in target_points_np:
        if draw_list[kp2[0]][kp2[1]] == 1:
            draw_list[kp2[0]][kp2[1]] = 3
        else:
            draw_list[kp2[0]][kp2[1]] = 2

    for i in range(20):
        for j in range(20):
            cprint("#", color_list[draw_list[i][j]], end=' ')
        print("")


def transform_points(points,i=0):
    if i!=0:
        random_index=i
    else:
        random_index = np.random.randint(8,size=1)[0]
    theta = random_index*np.pi/16
    T_matrix = [[np.cos(theta),-np.sin(theta)],
                [np.sin(theta),np.cos(theta)]]

    points_trans = np.dot(T_matrix, points.T)
    points_trans = points_trans.astype("int")

    return points_trans.T

def get_transform_angle(points,task_type):
    if "cloth" not in task_type:
        raise KeyError
    else:
        if "fold_a" not in task_type:
            theta = np.arctan((points[0][1]-points[4][1])/(points[4][0]-points[0][0]))
        else:
            theta = np.arctan((points[4][1]-points[8][1])/(points[8][0]-points[4][0]))
        if theta < 0:
            theta = theta+np.pi
        theta_list = np.abs(np.arange(8)*np.pi/16-theta)
        return np.argmin(theta_list)

def normalize_points(point_all,normalized_ratio,center_point=[0,0]):
    point_np = np.array(point_all)
    gemo_center_x = np.mean(point_np[:,0])
    gemo_center_y = np.mean(point_np[:,1])
    point_np = point_np - np.array([gemo_center_x,gemo_center_y])
    point_np = point_np*normalized_ratio
    point_np = point_np+np.array([center_point[0],center_point[1]])
    point_np = point_np.astype(int) 
    return  point_np

def get_iou(current_state,target_state):
    gemo_1 = MultiPoint(current_state).convex_hull
    gemo_2 = MultiPoint(target_state).convex_hull
    union_gemo = MultiPoint(np.concatenate((current_state,target_state))).convex_hull
    if not gemo_1.intersects(gemo_2):
        iou_value = 0
    else:
        inter_area = gemo_1.intersection(gemo_2).area
        union_area = union_gemo.area
        if union_area == 0:
            iou_value = 0
        else:
            iou_value=float(inter_area) / union_area
    return iou_value

def random_action(obj_state,env_type):

    if env_type == 0:
        total_length = len(obj_state)
        half_length = total_length//2

        current_state = obj_state[:half_length]
        target_state = obj_state[half_length:]

        differences_1 = target_state - current_state
        distances_1 = np.linalg.norm(differences_1, axis=1)
        average_distance_1 = np.mean(distances_1)

        differences_2 = target_state - current_state[::-1]
        distances_2 = np.linalg.norm(differences_2, axis=1)
        average_distance_2 = np.mean(distances_2)

        if average_distance_1 < average_distance_2:
            max_idx = np.argmax(distances_1)
            pick_pos = max_idx
            place_pos = max_idx
        else:
            max_idx = np.argmax(distances_2)
            pick_pos = half_length-1-max_idx
            place_pos = max_idx
    
    elif env_type == 1 or env_type == 2:   
        total_length = len(obj_state)
        half_length = total_length//2

        cur_state = obj_state[:half_length]
        tar_state = obj_state[half_length:]

        min_dist = np.float('inf')
        pick_pos, place_pos = None, None
        for a in range(total_length):
            if a < half_length:
                # mapping = [a, a+1, ..., num_parts-1, 0, 1, ..., a-1]
                mapping = [i for i in range(a, half_length)] + [i for i in range(0, a)]
            else:
                # Same as above but reverse it (to handle flipped ring).
                a -= half_length
                mapping = [i for i in range(a, half_length)] + [i for i in range(0, a)]
                mapping = mapping[::-1]
            differences = tar_state - cur_state[mapping]
            distances = np.linalg.norm(differences, axis=1)
            average_distance = np.mean(distances)

            if average_distance < min_dist:
                # Index of the largest distance among vertex + target.
                max_idx = np.argmax(distances)
                pick_pos = mapping[max_idx]
                place_pos = max_idx
                min_dist = average_distance

    return pick_pos,place_pos

def get_dis(current_state,target_state):
    half_length = len(current_state)
    total_length = 2*half_length
    min_dist = np.float('inf')
    for a in range(total_length):
        if a < half_length:
            # mapping = [a, a+1, ..., num_parts-1, 0, 1, ..., a-1]
            mapping = [i for i in range(a, half_length)] + [i for i in range(0, a)]
        else:
            # Same as above but reverse it (to handle flipped ring).
            a -= half_length
            mapping = [i for i in range(a, half_length)] + [i for i in range(0, a)]
            mapping = mapping[::-1]
        differences = target_state - current_state[mapping]
        distances = np.linalg.norm(differences, axis=1)
        average_distance = np.mean(distances)

        if average_distance < min_dist:
            min_dist = average_distance
    return min_dist

def random_action_rope(obj_state):
    total_length = len(obj_state)
    half_length = total_length//2

    current_state = obj_state[:half_length]
    target_state = obj_state[half_length:]

    differences_1 = target_state - current_state
    distances_1 = np.linalg.norm(differences_1, axis=1)
    average_distance_1 = np.mean(distances_1)

    differences_2 = target_state - current_state[::-1]
    distances_2 = np.linalg.norm(differences_2, axis=1)
    average_distance_2 = np.mean(distances_2)


    if average_distance_1 < average_distance_2:
        max_idx = np.argmax(distances_1)
        pick_pos = max_idx
        place_pos = max_idx
    else:
        max_idx = np.argmax(distances_2)
        pick_pos = half_length-1-max_idx
        place_pos = max_idx

    return pick_pos,place_pos

def get_dis_rope(current_state,target_state):

    differences_1 = target_state - current_state
    distances_1 = np.linalg.norm(differences_1, axis=1)
    average_distance_1 = np.mean(distances_1)

    differences_2 = target_state - current_state[::-1]
    distances_2 = np.linalg.norm(differences_2, axis=1)
    average_distance_2 = np.mean(distances_2)

    min_dist = min(average_distance_1,average_distance_2)
    return min_dist


def get_pybullet_quaternion_from_rot(rotation):
    """Abstraction for converting from a 3-parameter rotation to quaterion.

    This will help us easily switch which rotation parameterization we use.
    Quaternion should be in xyzw order for pybullet.

    Args:
      rotation: a 3-parameter rotation, in xyz order tuple of 3 floats
    Returns:
      quaternion, in xyzw order, tuple of 4 floats
    """
    euler_zxy = (rotation[2], rotation[0], rotation[1])
    quaternion_wxyz = transformations.quaternion_from_euler(*euler_zxy, axes='szxy')
    q = quaternion_wxyz
    quaternion_xyzw = (q[1], q[2], q[3], q[0])
    return quaternion_xyzw


def get_rot_from_pybullet_quaternion(quaternion_xyzw):
    """Abstraction for converting from quaternion to a 3-parameter toation.

    This will help us easily switch which rotation parameterization we use.
    Quaternion should be in xyzw order for pybullet.

    Args:
      quaternion, in xyzw order, tuple of 4 floats
    Returns:
      rotation: a 3-parameter rotation, in xyz order, tuple of 3 floats
    """
    q = quaternion_xyzw
    quaternion_wxyz = np.array([q[3], q[0], q[1], q[2]])
    euler_zxy = transformations.euler_from_quaternion(quaternion_wxyz, axes='szxy')
    euler_xyz = (euler_zxy[1], euler_zxy[2], euler_zxy[0])
    return euler_xyz




