import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import cv2 as cv
import utils

#rope:rope_line, rope_l, rope_v, rope_n
#ring:ring_circle, ring_square, move_ring
#cloth:cloth_fold, cloth_flatten, move_cloth

class rope_geometry(object):
    def __init__(self, geometry_type):
        part_instance = 8.56
        part_num = 24
        start_point = [59,119]
        self.keypoints = []
        self.geometry_type = geometry_type
        if self.geometry_type == 'rope_line':
            for i in range(part_num):
                self.keypoints.append([start_point[0]+part_instance*i,start_point[1]])
        elif self.geometry_type == 'rope_l':
            corner_point = 4+np.random.randint(3,size=1)[0]
            for i in range(corner_point):
                self.keypoints.append([start_point[0]+part_instance*i,start_point[1]])
            for j in range(part_num-corner_point):
                self.keypoints.append([start_point[0]+part_instance*(corner_point-1),start_point[1]+part_instance*j])
        elif self.geometry_type == 'rope_v':
            angle = (4+np.random.randint(3,size=1)[0])*10*np.pi/180
            for i in range(part_num//2):
                self.keypoints.append([start_point[0]+part_instance*i,start_point[1]])
            corner_point = [start_point[0]+part_instance*(part_num//2-1),start_point[1]]
            for j in range(part_num//2):
                self.keypoints.append([corner_point[0]-part_instance*np.cos(angle)*(j+1),corner_point[1]+part_instance*np.sin(angle)*(j+1)])
        elif self.geometry_type == 'rope_n':  
            angle = (4+np.random.randint(3,size=1)[0])*10*np.pi/180  
            for i in range(8):
                self.keypoints.append([start_point[0]+part_instance*i,start_point[1]])
            corner_point_1 = [start_point[0]+part_instance*7,start_point[1]]
            for j in range(8):
                self.keypoints.append([corner_point_1[0]-part_instance*np.cos(angle)*(j+1),corner_point_1[1]+part_instance*np.sin(angle)*(j+1)])
            corner_point_2 = [corner_point_1[0]-part_instance*np.cos(angle)*8,corner_point_1[1]+part_instance*np.sin(angle)*8]
            for p in range(8):
                self.keypoints.append([corner_point_2[0]+part_instance*(p+1),corner_point_2[1]])

class ring_geometry(object):
    def __init__(self, geometry_type):
        part_instance = 9
        part_num = 32
        start_point = [160,165]
        self.keypoints = []
        self.geometry_type = geometry_type
        if self.geometry_type == 'ring_circle':
            self.keypoints = [[160,165],[168,164],[177,162],[185,157],[192,152],[197,145],[202,137],[204,128],
                              [205,120],[204,111],[202,102],[197,94],[192,87],[185,82],[177,77],[168,75],
                              [160,74],[151,75],[142,77],[134,82],[127,87],[122,94],[117,102],[115,111],
                              [114,120],[115,128],[117,137],[122,145],[127,152],[134,157],[142,162],[151,164]]

        elif self.geometry_type == 'ring_square':
            self.keypoints.append(start_point)
            for i in range(part_num//4):
                self.keypoints.append([start_point[0]+part_instance*(i+1),start_point[1]])
            corner_point_1 = [start_point[0]+part_instance*8,start_point[1]]    
            for i in range(part_num//4):
                self.keypoints.append([corner_point_1[0],corner_point_1[1]-part_instance*(i+1)])
            corner_point_2 = [start_point[0]+part_instance*8,start_point[1]-part_instance*8]  
            for i in range(part_num//4):
                self.keypoints.append([corner_point_2[0]-part_instance*(i+1),corner_point_2[1]])
            corner_point_3 = [start_point[0],start_point[1]-part_instance*8]
            for i in range(part_num//4-1):
                self.keypoints.append([corner_point_3[0],corner_point_3[1]+part_instance*(i+1)])

        elif self.geometry_type == 'ring_move':
            self.keypoints = [[160,165],[168,164],[177,162],[185,157],[192,152],[197,145],[202,137],[204,128],
                              [205,120],[204,111],[202,102],[197,94],[192,87],[185,82],[177,77],[168,75],
                              [160,74],[151,75],[142,77],[134,82],[127,87],[122,94],[117,102],[115,111],
                              [114,120],[115,128],[117,137],[122,145],[127,152],[134,157],[142,162],[151,164]]

class cloth_geometry(object):
    def __init__(self, geometry_type):
        self.geometry_type = geometry_type
        self.origin_keypoints = np.array([[99,59],[99,89],[99,120],[99,150],
                                         [99,180],[129,180],[159,180],[190,180],
                                         [220,180],[220,150],[220,119],[220,89],
                                         [220,59],[190,59],[159,59],[129,59]])
        if self.geometry_type == 'cloth_flatten':
            self.mappings = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]                       
        elif self.geometry_type == 'cloth_fold':  
            self.mappings = [0,1,2,3,4,5,6,5,4,3,2,1,0,15,14,15]   
        elif self.geometry_type == 'cloth_fold_a':
            self.mappings = [12,13,14,15,0,1,2,3,4,3,2,1,0,15,14,13]     
        self.keypoints = self.origin_keypoints[self.mappings]




class point_geometry(object):
    def __init__(self, point_num, geometry_type):
        self.loc_min = 0
        self.loc_max = 100
        self.geometry_type = geometry_type
        self.point_num = point_num
        self.keypoints = []
        self.edge_length = []
        self.point_all = []
        if self.geometry_type==1:
            self.num_part = 32
            self.remain_num = 32
        else:
            self.num_part = 20
            self.remain_num = 20
        for i in range(self.point_num):
            x_location = np.random.randint(self.loc_min, self.loc_max,1)[0]
            y_location = np.random.randint(self.loc_min, self.loc_max,1)[0]
            self.keypoints.append([x_location,y_location])
        if self.geometry_type==1:
            edge_num = self.point_num
        else:
            edge_num = self.point_num-1
        for i in range(edge_num):
            p_0 = i
            p_1 = (i+1)%self.point_num
            dis = math.sqrt((self.keypoints[p_1][0]-self.keypoints[p_0][0])**2+(self.keypoints[p_1][1]-self.keypoints[p_0][1])**2 )
            self.edge_length.append(dis)
        all_length = np.sum(self.edge_length)
        for i in range(edge_num):
            self.point_all.append(self.keypoints[i])
            if i == edge_num-1:
                insert_num = self.remain_num
            else:
                insert_num = int(self.num_part*self.edge_length[i]/all_length+0.5)  #4 and 5 
            for j in range(insert_num-1):
                insert_point = copy.deepcopy(self.keypoints[i])
                next_loc = (i+1)%self.point_num
                insert_point[0] += (j+1)*(self.keypoints[next_loc][0] - self.keypoints[i][0])/insert_num
                insert_point[1] += (j+1)*(self.keypoints[next_loc][1] - self.keypoints[i][1])/insert_num
                self.point_all.append(insert_point)
            self.remain_num = self.num_part-len(self.point_all)
        

  

    def is_available(self):
        if np.min(self.edge_length) == 0:
            return False
        else:
            check_ratio = np.max(self.edge_length)/np.min(self.edge_length)
            if self.geometry_type==1:
                convex_0 = utils.isConvex(self.keypoints)
                convex_1 = utils.isAngle_r(self.keypoints)
                convex = convex_0 and convex_1
            else:
                convex_0 = utils.isCross(self.keypoints)
                convex_1 = utils.isAngle(self.keypoints)
                convex = convex_0 and convex_1
            if check_ratio<2 and convex:
                if len(self.point_all) != self.num_part:
                    print(self.point_all)
                    print("error")
                    return False
                else:
                    return True
            else:
                return False

    def geometry_normalize(self,normalized_num,center_point=[0,0],is_show=False):
        normalized_ratio = normalized_num/np.sum(self.edge_length)
        point_np = utils.normalize_points(self.point_all,normalized_ratio,center_point)

        #plt.savefig('gemo.jpg')
        if is_show:
            plt.scatter(point_np[:,0],point_np[:,1])
            plt.show()
        return point_np
             


if __name__ == "__main__":
    my_gemo = point_geometry(4)
    print("check_resu:",my_gemo.is_available())
    my_gemo.geometry_normalize(normalized_num=1000,center_point=[20,0],is_show=True)