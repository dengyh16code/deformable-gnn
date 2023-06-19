#!/usr/bin/env python

import os
import sys
import time
import threading
import pkg_resources

import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from gripper import Suction
import utils
import cameras
import cv2 as cv
import math



class Environment():

    def __init__(self, env_type=0, disp=False, hz=480):
        """Creates OpenAI gym-style env with support for PyBullet threading.

        Args:
            disp: Whether or not to use PyBullet's built-in display viewer.
                Use this either for local inspection of PyBullet, or when
                using any soft body (cloth or bags), because PyBullet's
                TinyRenderer graphics (used if disp=False) will make soft
                bodies invisible.
            hz: Parameter used in PyBullet to control the number of physics
                simulation steps. Higher values lead to more accurate physics
                at the cost of slower computaiton time. By default, PyBullet
                uses 240, but for soft bodies we need this to be at least 480
                to avoid cloth intersecting with the plane.
        """
        self.ee = None
        self.objects = []
        self.running = False
        self.fixed_objects = []
        self.cable_bead_IDs = []
        self.env_type = env_type #env_type 0:rope,1:rope_ring,2:cloth
        self.def_IDs = []
        self.index_angle = 0

        
        self.pixel_size = 0.001644#0.001096
        self.camera_config = cameras.RealSenseD415.CONFIG
        self.camera_config_up = cameras.RealSenseD415.CONFIG_UP

        self.homej = np.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self._IDs = {}

         
        self.t_lim = 15 # Set default movej timeout limit.
        self.hz = hz
        self.colors = [utils.COLORS['blue']]


        # Start PyBullet.
        p.connect(p.GUI if disp else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        assets_path = os.path.dirname(os.path.abspath(__file__))
        p.setAdditionalSearchPath(assets_path)

        # Check PyBullet version (see also the cloth/bag task scripts!).
        p_version = pkg_resources.get_distribution('pybullet').version
        tested = ['2.8.4', '3.0.4']
        assert p_version in tested, f'PyBullet version {p_version} not in {tested}'

        # Move the camera a little closer to the scene. Most args are not used.
        # PyBullet defaults: yaw=50 and pitch=-35.
        if disp:
            _, _, _, _, _, _, _, _, _, _, _, target = p.getDebugVisualizerCamera()
            p.resetDebugVisualizerCamera(
                cameraDistance=0.9,
                cameraYaw=90,
                cameraPitch=-30,
                cameraTargetPosition=target,)

        # Control PyBullet simulation steps.
        self.step_thread = threading.Thread(target=self.step_simulation)
        self.step_thread.daemon = True
        self.step_thread.start()


    def step_simulation(self):
        """Adding optional hertz parameter for better cloth physics.

        From our discussion with Erwin, we should just set time.sleep(0.001),
        or even consider removing it all together. It's mainly for us to
        visualize PyBullet with the GUI to make it not move too fast
        """
        p.setTimeStep(1.0 / self.hz)
        while True:
            if self.running:
                p.stepSimulation()
            if self.ee is not None:
                self.ee.step()
            time.sleep(0.001)

    def stop(self):
        p.disconnect()
        del self.step_thread

    def start(self):
        self.running = True

    def pause(self):
        self.running = False
    

    def remove_deform(self):
        if self.env_type == 0 or self.env_type == 1:
            for part_id in self.cable_bead_IDs:
                #self._IDs[part_id] = f'cable_part_{str(part_id).zfill(2)}'
                self.objects.remove(part_id)
                p.removeBody(part_id)
            self.cable_bead_IDs = []
            #self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)


    def add_cable_ring(self,center_pos = None):

        self.num_parts = 32
        self.radius = 0.005
        self.ring_radius = 0.075
        self.targets_visible = True
        self.object_points = {}
        self.primitive_params = {
                'speed': 0.001,
                'delta_z': -0.001,
                'postpick_z': 0.04,
                'preplace_z': 0.04,
                'pause_place': 0.0,    
        }  #pick and place param

        def rad_to_deg(rad):
            return (rad * 180.0) / np.pi

        def get_discretized_rotations(num_rotations):
            # counter-clockwise
            theta = i * (2 * np.pi) / num_rotations
            return (theta, rad_to_deg(theta))

        # Bead properties.
        #color = self.colors[0] + [1]
        color= [0.8, 0.6, 0.4,1]

        beads = []
        bead_positions_l = []

        # Add beaded cable. Here, `position` is the circle center.
        position = np.float32([0.5,0])
        if center_pos is not None:
            position = position+center_pos[:2]
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.radius]*3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=self.radius*1.5)

        # Iterate through parts and create constraints as needed.
        for i in range(self.num_parts):
            angle_rad, _ = get_discretized_rotations(self.num_parts)
            px = self.ring_radius * np.cos(angle_rad)
            py = self.ring_radius * np.sin(angle_rad)
            #print(f'pos: {px:0.2f}, {py:0.2f}, angle: {angle_rad:0.2f}, {angle_deg:0.1f}')
            bead_position = np.float32([position[0] + px, position[1] + py, 0.01])
            part_id = p.createMultiBody(0.1, part_shape, part_visual,
                    basePosition=bead_position)
            p.changeVisualShape(part_id, -1, rgbaColor=color)

            if i > 0:
                parent_frame = bead_position - bead_positions_l[-1]
                constraint_id = p.createConstraint(
                        parentBodyUniqueId=beads[-1],
                        parentLinkIndex=-1,
                        childBodyUniqueId=part_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_POINT2POINT,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=parent_frame,
                        childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)

            # Make a constraint with i=0. Careful with `parent_frame`!
            if i == self.num_parts- 1:
                parent_frame = bead_positions_l[0] - bead_position
                constraint_id = p.createConstraint(
                        parentBodyUniqueId=part_id,
                        parentLinkIndex=-1,
                        childBodyUniqueId=beads[0],
                        childLinkIndex=-1,
                        jointType=p.JOINT_POINT2POINT,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=parent_frame,
                        childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)

            # Track beads.
            beads.append(part_id)
            bead_positions_l.append(bead_position)

            # The usual for tracking IDs. Four things to add.max_force
            self.cable_bead_IDs.append(part_id)
            self._IDs[part_id] = f'cable_part_{str(part_id).zfill(2)}'
            self.objects.append(part_id)
            self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)

    def add_cable(self, direction='y',):

        self.num_parts = 24
        self.radius = 0.005
        self.length = 2 * self.radius * self.num_parts * np.sqrt(2)
        #color = self.colors[0] + [1]
        #color_end = utils.COLORS['blue'] + [1]
        color= [0.8, 0.6, 0.4,1]
        self.object_points = {}
        self.primitive_params = {
                    'speed': 0.001,
                    'delta_z': -0.001,
                    'postpick_z': 0.04,
                    'preplace_z': 0.04,
                    'pause_place': 0.0,       
                    }

        # Add beaded cable.
        distance = self.length / self.num_parts
        position = np.float32([0.5,-0.18,0])
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.radius]*3)
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=self.radius*1.5)

        # Iterate through parts and create constraints as needed.
        for i in range(self.num_parts):
            if direction == 'x':
                position[0] += distance
                parent_frame = (distance, 0, 0)
            elif direction == 'y':
                position[1] += distance
                parent_frame = (0, distance, 0)
            else:
                position[2] += distance
                parent_frame = (0, 0, distance)

            part_id = p.createMultiBody(0.1, part_shape, part_visual,
                    basePosition=position)
            if i > 0:
                constraint_id = p.createConstraint(
                        parentBodyUniqueId=self.objects[-1],
                        parentLinkIndex=-1,
                        childBodyUniqueId=part_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_POINT2POINT,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=parent_frame,
                        childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=100)

            '''
            if (i > 0) and (i < self.num_parts - 1):
                p.changeVisualShape(part_id, -1, rgbaColor=color)
            elif i == self.num_parts - 1:
                p.changeVisualShape(part_id, -1, rgbaColor=color_end)
            '''

            p.changeVisualShape(part_id, -1, rgbaColor=color)

            # The usual for tracking IDs. Four things to add.
            self.cable_bead_IDs.append(part_id)
            self._IDs[part_id] = f'cable_part_{str(part_id).zfill(2)}'
            self.objects.append(part_id)
            self.object_points[part_id] = np.float32((0, 0, 0)).reshape(3, 1)


    def add_cloth(self, give_angle = None):
        """Adding a cloth from an .obj file."""
        self.n_cuts = 5
        self._cloth_scale = 0.1
        self._f_cloth = "/plane/bl_cloth_05_cuts.obj"
        self.corner_indices = [0, 20, 24, 4]  # actual corners
        self.object_points = {}

        self.primitive_params = {
                'speed': 0.002,
                'delta_z': -0.0010,
                'postpick_z': 0.05,
                'preplace_z': 0.05,
                'pause_place': 0.0,      
        }

        self._mass = 0.5
        self._edge_length = (2.0 * self._cloth_scale) / (self.n_cuts - 1)
        self._collisionMargin = self._edge_length / 5.0
        self._cloth_length = (2.0 * self._cloth_scale)
        self._cloth_size = (self._cloth_length, self._cloth_length, 0.01)

        #Sample a flat cloth arbitrarily on the workspace.
        self.index_angle = np.random.randint(8,size=1)[0]
        if give_angle is not None:
            self.index_angle = give_angle
        orn_random = [np.pi / 2.0,0,self.index_angle*np.pi/16] #np.random.normal(loc=0.0, scale=0.5)
        #orn_random = [np.pi / 2.0,0,0]
        base_orn = p.getQuaternionFromEuler(orn_random)
        base_pos= np.float32([0.5,0,0.01])

        self.cloth_id = p.loadSoftBody(
                fileName=self._f_cloth,
                basePosition=base_pos,
                baseOrientation=base_orn,
                collisionMargin=self._collisionMargin,
                scale=self._cloth_scale,
                mass=self._mass,
                useNeoHookean=0,
                useBendingSprings=1,
                useMassSpring=1,
                springElasticStiffness=40,
                springDampingStiffness=0.1,
                springDampingAllDirections=0,
                useSelfCollision=1,
                frictionCoeff=1.0,
                useFaceContact=1,)

        # Only if using more recent PyBullet versions.
        p_version = pkg_resources.get_distribution('pybullet').version
        if p_version == '3.0.4':
            color = [0.8, 0.6, 0.4,1] #self.colors[0] + [1]
            p.changeVisualShape(self.cloth_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
                                rgbaColor=color)

        # For tracking IDs and consistency with existing ravens code.
        self._IDs[self.cloth_id] = 'cloth'
        self.object_points[self.cloth_id] = np.float32((0, 0, 0)).reshape(3, 1)

        # To help environment pick-place method track all deformables.
        self.def_IDs.append(self.cloth_id)
 
        # Sanity checks.
        nb_vertices, _ = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        assert nb_vertices == self.n_cuts * self.n_cuts
       


    #-------------------------------------------------------------------------
    # Standard RL Functions
    #-------------------------------------------------------------------------

    def step(self, pick_point,place_point):
        rgb,depth = self.render(self.camera_config_up)

        pick_point_tf = utils.pixel_to_position(pick_point,depth[pick_point[1]][pick_point[0]],self.camera_config_up,self.pixel_size)
        place_point_tf = utils.pixel_to_position(place_point,depth[place_point[1]][place_point[0]],self.camera_config_up,self.pixel_size)
        pick_point_add_ori = [pick_point_tf,[0,0,0,1]]
        place_point_add_ori = [place_point_tf,[0,0,0,1]]
        #print("pick_point:",pick_point_add_ori)
        #print("place_point:",place_point_add_ori)
        #print("...............")
        self.pick_place(pick_point_add_ori,place_point_add_ori) 
        return pick_point_tf,place_point_tf
        


    def reset(self, center_pos =None, give_angle = None, disable_render_load=True):
        """Sets up PyBullet, loads models

        Args:
            disable_render_load: Need this as True to avoid `p.loadURDF`
                becoming a time bottleneck, judging from my profiling.
        """
        self.pause()
        self.objects = []
        self.fixed_objects = []

        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)   
        p.setGravity(0, 0, -9.8)

        # Empirically, this seems to make loading URDFs faster w/remote displays.
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        id_plane = p.loadURDF(f'plane/plane.urdf', [0, 0, -0.001])
        id_ws = p.loadURDF(f'table/table.urdf', [0.5, 0, 0])
        #id_ws = p.loadURDF(f'ur5/workspace.urdf', [0.5, 0, 0])

        # Load UR5 robot arm equipped with task-specific end effector.
        self.ur5 = p.loadURDF(f'ur5/ur5-suction.urdf')
        self.ee_tip_link = 12
        self.ee = Suction(self.ur5, 11)

        if self.env_type == 0:
            self.add_cable()
        elif self.env_type == 1:
            self.add_cable_ring(center_pos)
        elif self.env_type == 2:
            self.add_cloth(give_angle)
            self.ee.set_def_threshold(threshold=0.025)
            self.ee.set_def_nb_anchors(nb_anchors=1)

        # Get revolute joint indices of robot (skip fixed joints).
        num_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(num_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])

        # Get end effector tip pose in home configuration.
        ee_tip_state = p.getLinkState(self.ur5, self.ee_tip_link)
        self.home_pose = np.array(ee_tip_state[0] + ee_tip_state[1])

        # Reset end effector.
        self.ee.release()

        # Setting for deformable object
        #self.ee.set_def_threshold(threshold=self.task.def_threshold)
        #self.ee.set_def_nb_anchors(nb_anchors=self.task.def_nb_anchors)
        assert self.hz >= 480, f'Error, hz={self.hz} is too small!'

        # Restart simulation.
        self.start()
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        self.get_current_state()

    
    def random_initialize(self,random_times=1):
        pick_random,place_random = 0,0
        for random in range(random_times):
                now_state = self.get_current_state()   #get observation
                object_length = len(now_state) //2 
                pick_random = np.random.randint(0, object_length)
                place_random = np.random.randint(0, object_length)
                pick_point = now_state[pick_random]
                place_point = now_state[place_random]
                self.step(pick_point,place_point)
        print("random finish")
        return pick_random,place_random

    def render(self,config):
        """Render RGB-D image with specified configuration."""
        # Compute OpenGL camera settings.
        lookdir = np.array([0, 0, 1]).reshape(3, 1)
        updir = np.array([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config['rotation'])
        rotm = np.array(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_length = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = p.computeViewMatrix(config['position'], lookat, updir)
        fovh = (np.arctan((config['image_size'][0] /
                           2) / focal_length) * 2 / np.pi) * 180
        #self.pixel_size=  0.5 * (config['position'][2] / np.tan(fovh * np.pi / 360)) /config['image_size'][0]

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config['image_size'][1] / config['image_size'][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=config['image_size'][1],
            height=config['image_size'][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (config['image_size'][0],
                            config['image_size'][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        color_image_size = (color_image_size[0], color_image_size[1], 3)
        if config['noise']:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color_image_size))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config['image_size'][0], config['image_size'][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        if config['noise']:
            depth += np.random.normal(0, 0.003, depth_image_size)


        return color, depth


    #-------------------------------------------------------------------------
    # Record Functions
    #-------------------------------------------------------------------------

    def record_video(self,file_name,save_dir):

        dir_name = os.path.join('record/',save_dir)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.curr_recording= p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                os.path.join(dir_name, '{}.mp4'.format(file_name)))
    
    def stop_record(self):
         p.stopStateLogging(self.curr_recording)


    #-------------------------------------------------------------------------
    # Robot Movement Functions
    #-------------------------------------------------------------------------

    def movej(self, targj, speed=0.01, t_lim=20):
        """Move UR5 to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < t_lim:
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)
            time.sleep(0.001)
        print('Warning: movej exceeded {} sec timeout. Skipping.'.format(t_lim))
        return False

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        # # Keep joint angles between -180/+180
        # targj[5] = ((targj[5] + np.pi) % (2 * np.pi) - np.pi)
        targj = self.solve_IK(pose)
        return self.movej(targj, speed, self.t_lim)

    def solve_IK(self, pose):
        homej_list = np.array(self.homej).tolist()
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip_link,
            targetPosition=pose[:3],
            targetOrientation=pose[3:],
            lowerLimits=[-17, -2.3562, -17, -17, -17, -17],
            upperLimits=[17, 0, 17, 17, 17, 17],
            jointRanges=[17] * 6,
            restPoses=homej_list,
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.array(joints)
        joints[joints > 2 * np.pi] = joints[joints > 2 * np.pi] - 2 * np.pi
        joints[joints < -2 * np.pi] = joints[joints < -2 * np.pi] + 2 * np.pi
        return joints

    #-------------------------------------------------------------------------
    # Motion Primitives
    #-------------------------------------------------------------------------

    def pick_place(self, pose0, pose1,save_index = 1000):
        """Execute pick and place primitive.

        Standard ravens tasks use the `delta` vector to lower the gripper
        until it makes contact with something. With deformables, however, we
        need to consider cases when the gripper could detect a rigid OR a
        soft body (cloth or bag); it should grip the first item it touches.
        This is handled in the Gripper class.

        Different deformable ravens tasks use slightly different parameters
        for better physics (and in some cases, faster simulation). Therefore,
        rather than make special cases here, those tasks will define their
        own action parameters, which we use here if they exist. Otherwise, we
        stick to defaults from standard ravens. Possible action parameters a
        task might adjust:

            speed: how fast the gripper moves.
            delta_z: how fast the gripper lowers for picking / placing.
            prepick_z: height of the gripper when it goes above the target
                pose for picking, just before it lowers.
            postpick_z: after suction gripping, raise to this height, should
                generally be low for cables / cloth.
            preplace_z: like prepick_z, but for the placing pose.
            pause_place: add a small pause for some tasks (e.g., bags) for
                slightly better soft body physics.
            final_z: height of the gripper after the action. Recommended to
                leave it at the default of 0.3, because it has to be set high
                enough to avoid the gripper occluding the workspace when
                generating color/depth maps.
        Args:
            pose0: picking pose.
            pose1: placing pose.

        Returns:
            A bool indicating whether the action succeeded or not, via
            checking the sequence of movep calls. If any movep failed, then
            self.step() will terminate the episode after this action.
        """
        #initial params
        speed = 0.01
        delta_z = -0.001
        prepick_z = 0.3
        postpick_z = 0.3
        preplace_z = 0.3
        pause_place = 0.0
        final_z = 0.3

        # adjust action params
        speed       = self.primitive_params['speed']
        delta_z     = self.primitive_params['delta_z']
        postpick_z  = self.primitive_params['postpick_z']
        preplace_z  = self.primitive_params['preplace_z']
        pause_place = self.primitive_params['pause_place']

        # Used to track deformable IDs, so that we can get the vertices.
        def_IDs = self.def_IDs #self.cable_bead_IDs

        # Otherwise, proceed as normal.
        success = True
        pick_position = np.array(pose0[0])
        pick_rotation = np.array(pose0[1])
        prepick_position = pick_position.copy()
        prepick_position[2] = prepick_z

        # Execute picking motion primitive.
        prepick_pose = np.hstack((prepick_position, pick_rotation))
        success &= self.movep(prepick_pose)
        target_pose = prepick_pose.copy()
        delta = np.array([0, 0, delta_z, 0, 0, 0, 0])

        # Lower gripper until (a) touch object (rigid OR softbody), or (b) hit ground.
        while not self.ee.detect_contact(def_IDs) and target_pose[2] > 0:
            target_pose += delta
            success &= self.movep(target_pose)

        # Create constraint (rigid objects) or anchor (deformable).
        self.ee.activate(self.objects, def_IDs)

        # Increase z slightly (or hard-code it) and check picking success.

        prepick_pose[2] = postpick_z
        success &= self.movep(prepick_pose, speed=speed)
        time.sleep(pause_place) # extra rest for bags
        '''
        elif isinstance(self.task, tasks.names['cable']):
            prepick_pose[2] = 0.03
            success &= self.movep(prepick_pose, speed=0.001)
        else:
            prepick_pose[2] += pick_position[2]
            success &= self.movep(prepick_pose)
        '''
        
        pick_success = self.ee.check_grasp()

        if pick_success:
            place_position = np.array(pose1[0])
            place_rotation = np.array(pose1[1])
            preplace_position = place_position.copy()
            preplace_position[2] = 0.3 + pick_position[2]

            # Execute placing motion primitive if pick success.
            preplace_pose = np.hstack((preplace_position, place_rotation))
            preplace_pose[2] = preplace_z
            success &= self.movep(preplace_pose, speed=speed)
            time.sleep(pause_place) # extra rest for bags
            
            
            '''
            rgb,depth = self.render(cameras.RealSenseD415.CONFIG[0])
            file_name = str(save_index)+'.jpg'
            rgb = cv.cvtColor(rgb,cv.COLOR_BGR2RGB)
            print("save it")
            cv.imwrite("record/"+file_name,rgb)
            '''
            
            
            '''
            elif isinstance(self.task, tasks.names['cable']):
                preplace_pose[2] = 0.03
                success &= self.movep(preplace_pose, speed=0.001)
            else:
                success &= self.movep(preplace_pose)
            '''

            # Lower the gripper. Here, we have a fixed speed=0.01. TODO: consider additional
            # testing with bags, so that the 'lowering' process for bags is more reliable.
            target_pose = preplace_pose.copy()
            while not self.ee.detect_contact(def_IDs) and target_pose[2] > 0:
                target_pose += delta
                success &= self.movep(target_pose)

            # Release AND get gripper high up, to clear the view for images.
            self.ee.release()
            preplace_pose[2] = final_z
            success &= self.movep(preplace_pose)
        else:
            # Release AND get gripper high up, to clear the view for images.
            self.ee.release()
            prepick_pose[2] = final_z
            success &= self.movep(prepick_pose)
        # Move robot to home joint configuration.
        initial_pos = np.array([0, 0.487, 0.32,0,0,0,1])
        success_initial = self.movep(initial_pos)
        if not success_initial:
            # Move robot to home joint configuration.
            for i in range(len(self.joints)):
                p.resetJointState(self.ur5, self.joints[i], self.homej[i])


        return success

    def sweep(self, pose0, pose1):
        """Execute sweeping primitive."""
        success = True
        position0 = np.float32(pose0[0])
        position1 = np.float32(pose1[0])
        direction = position1 - position0
        length = np.linalg.norm(position1 - position0)
        if length == 0:
            direction = np.float32([0, 0, 0])
        else:
            direction = (position1 - position0) / length

        theta = np.arctan2(direction[1], direction[0])
        rotation = p.getQuaternionFromEuler((0, 0, theta))

        over0 = position0.copy()
        over0[2] = 0.3
        over1 = position1.copy()
        over1[2] = 0.3

        success &= self.movep(np.hstack((over0, rotation)))
        success &= self.movep(np.hstack((position0, rotation)))

        num_pushes = np.int32(np.floor(length / 0.01))
        for i in range(num_pushes):
            target = position0 + direction * num_pushes * 0.01
            success &= self.movep(np.hstack((target, rotation)), speed=0.003)

        success &= self.movep(np.hstack((position1, rotation)), speed=0.003)
        success &= self.movep(np.hstack((over1, rotation)))
        return success

    def push(self, pose0, pose1):
        """Execute pushing primitive."""
        p0 = np.float32(pose0[0])
        p1 = np.float32(pose1[0])
        p0[2], p1[2] = 0.025, 0.025
        if np.sum(p1 - p0) == 0:
            push_direction = 0
        else:
            push_direction = (p1 - p0) / np.linalg.norm((p1 - p0))
        p1 = p0 + push_direction * 0.01
        success = True
        success &= self.movep(np.hstack((p0, self.home_pose[3:])))
        success &= self.movep(np.hstack((p1, self.home_pose[3:])), speed=0.003)
        return success


    #-------------------------------------------------------------------------
    # state function
    #-------------------------------------------------------------------------

    def get_current_state(self):
        """get current state of the deformable rope from the env
        """
        rgb,depth = self.render(self.camera_config_up)
        self.cable_pos_l = []
        self.cable_pixel_l = []
        self.normalize_length = 0
        self.cable_center = np.array([0,0])
        if self.env_type < 2:  #1 degree
            for bead_ID in self.cable_bead_IDs:
                bead_position = p.getBasePositionAndOrientation(bead_ID)[0]
                self.cable_pos_l.append(bead_position)
                cal_pixel = utils.position_to_pixel(bead_position,self.camera_config_up,self.pixel_size)
                self.cable_pixel_l.append(cal_pixel)           
            if self.env_type==0:
                edge_num = self.num_parts-1
            elif self.env_type==1:
                edge_num = self.num_parts
            for i in range(edge_num):
                p_0 = i
                p_1 = (i+1)%self.num_parts
                dis = math.sqrt((self.cable_pixel_l[p_1][0]-self.cable_pixel_l[p_0][0])**2+(self.cable_pixel_l[p_1][1]-self.cable_pixel_l[p_0][1])**2 )
                self.normalize_length += dis
                                   
            #self.normalize_length = math.sqrt((self.cable_pixel_l[10][0]-self.cable_pixel_l[9][0])**2+(self.cable_pixel_l[10][1]-self.cable_pixel_l[9][1])**2)
        else:
            _, vert_poses = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
            for vert_pos in vert_poses:
                self.cable_pos_l.append(vert_pos)
                cal_pixel = utils.position_to_pixel(vert_pos,self.camera_config_up,self.pixel_size)
                self.cable_pixel_l.append(cal_pixel)
    
            self.normalize_length = self._edge_length

    
        current_state = np.array(self.cable_pixel_l)
        if self.env_type == 2:
            current_state = current_state[[0,1,2,3,4,9,14,19,24,23,22,21,20,15,10,5]]

        cabel_center_x = np.mean(current_state[:,0])
        cabel_center_y = np.mean(current_state[:,1])
        self.cable_center = [cabel_center_x,cabel_center_y]
        return current_state
