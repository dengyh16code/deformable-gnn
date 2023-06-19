#!/usr/bin/env python

import numpy as np
import pybullet as p


class RealSenseD415():
    """Default configuration with 3 RealSense RGB-D cameras."""
    
 
    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (240, 320)  #480 640
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

    # Set default camera poses.
    render_position = (1.8, 0, 1.35) # (1,0,0.75)
    render_rotation = (np.pi / 3.5, np.pi, -np.pi / 2)#(np.pi / 4, np.pi, -np.pi / 2)
    render_rotation = p.getQuaternionFromEuler(render_rotation)

    front_position = (1., 0, 0.75)
    front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    left_position = (0, 0.5, 0.75)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)
    right_position = (0, -0.5, 0.75)
    right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)

    up_position = (0.5, 0, 0.75)  #0.5
    up_rotation = (0, np.pi, -np.pi/2)
    up_rotation = p.getQuaternionFromEuler(up_rotation)

    CONFIG_UP = {
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': up_position,
            'rotation': up_rotation,
            'zrange': (0.01, 10.),
            'noise': False
              }

    CONFIG_RENDER = {
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': render_position,
            'rotation': render_rotation,
            'zrange': (0.01, 10.),
            'noise': False
               }

    CONFIG = [
        {
            'image_size': (480,640),
            'intrinsics': intrinsics,
            'position': front_position,
            'rotation': front_rotation,
            'zrange': (0.01, 10.),
            'noise': False
        },
        {
            'image_size': (480,640),
            'intrinsics': intrinsics,
            'position': left_position,
            'rotation': left_rotation,
            'zrange': (0.01, 10.),
            'noise': False
        },
        {
            'image_size': (480,640),
            'intrinsics': intrinsics,
            'position': right_position,
            'rotation': right_rotation,
            'zrange': (0.01, 10.),
            'noise': False
        }]
