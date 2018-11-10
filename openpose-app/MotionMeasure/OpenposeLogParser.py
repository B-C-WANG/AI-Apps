# encoding: utf-8
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import queue
import _thread
import traceback



point_name = [
 "Nose",
  "Neck",
  "RShoulder",
  "RElbow",
  "RWrist",
  "LShoulder",
  "LElbow",
  "LWrist",
  "MidHip",
  "RHip",
"RKnee",
"RAnkle",
"LHip",
"LKnee",
"LAnkle",
"REye",
"LEye",
"REar",
"LEar",
"LBigToe",
"LSmallToe",
"LHeel",
"RBigToe",
"RSmallToe",
"RHeel",
"Background"
]



class OpenposeJsonParser():
    def __init__(self):
        pass

    def get_pose2d_state_of_first_people(self,json_file):
        '''

        get all body points of first people:

        - all points will  minus the position of point 1 to set 1 as center
        (1 is "Neck")
        to avoid the change of distance to camera
        the distance of two points will be scaled by distance of 2-5
        2-5 is RShoulder and LShoulder, the distance can not change much with body

        in this way, all point distance is use 1 as center
        and use distance of 2-5 as 1,
        so it can be used to compare between two frame










        '''
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            people = data["people"]

            people = people[0]


            pose2d = people["pose_keypoints_2d"]
            pose2d = np.array(pose2d).reshape(-1,3)
            #print(pose2d) # x y and confidence
            coord = pose2d[:,:2]
            center_pos = coord[1]
            if (center_pos[0] < [0.1,0.1]).any():
                # return false if can not detect center point
                return False
            if (coord[2] < [0.1,0.1]).any() or (coord[5] < [0.1,0.1]).any():
                # return false if can not detect 2 5 point
                return False


            # set the position of [0,0] to center position so that will be 9 after minus
            coord[(coord[:,:2] <[0.1,0.1]).any(axis=1)] = center_pos

            # set center position
            coord = coord - center_pos

            # reset
            coord = - coord

            # scale according to refer_distance
            refer_distance = np.sqrt(np.sum(np.square(coord[2]-coord[5])))
            #print(refer_distance)
            coord = coord / refer_distance
            data ={}
            #print(coord)

            for i in range(coord.shape[0]):
                if (np.abs(coord[i,:]) < ([0.0001,0.0001])).any():
                    data[point_name[i]] = False
                else:
                    data[point_name[i]] = coord[i,:]
            # finally add center_position, this center_position is the total_move of the body, its absolute value is meanningless
            data[point_name[1]] = center_pos / refer_distance


            return data
        except:
            # if met error, all set False
            info = {}
            for i in point_name[:-1]:
                info[i] = False
            return info


    def get_point_change_data(self,last_state,now_state,sum=False):
        try:
            # all points move distance, related to energy people use
            info = {}

            for name in point_name[:-1]:# background not included
                    if now_state is bool:
                        raise ValueError()
                    if isinstance(now_state[name],bool)  or  isinstance(last_state[name],bool):
                        info[name] = False
                    else:
                        info[name] = np.sqrt(np.sum(np.square(last_state[name] - now_state[name])))
            if sum == False:

                return info
            else:
                value = 0
                for i in info:
                    value += 0 if info[i] == False else abs(info[i])
                return value
        except:
            traceback.print_exc()
            if sum==False:
                info = {}
                for i in point_name:
                    info[i] = 0
                return  info
            else:
                return 0



    def stream_update_point_change_data_in_the_dir(self,json_file_dir,sum=False):
        last_state = None


        while last_state is None:

            file = os.listdir(json_file_dir)
            for i in file:
                if i.endswith(".json"):
                    file_path = json_file_dir + "/" + i
                    last_state = self.get_pose2d_state_of_first_people(file_path)
                    os.remove(file_path)
                    break
        while 1:
            file = os.listdir(json_file_dir)
            for i in file:
                if i.endswith(".json"):
                    file_path = json_file_dir + "/" + i
                    now_state = self.get_pose2d_state_of_first_people(file_path)
                    os.remove(file_path)
                    yield self.get_point_change_data(last_state,now_state,sum=sum)
                    break









if __name__ == '__main__':



    while 1:
        for i in  OpenposeJsonParser().stream_update_point_change_data_in_the_dir("G:\openpose\output",sum=True):
            print(i)



