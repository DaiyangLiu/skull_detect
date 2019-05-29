import numpy as np
import copy


class skull_detector():
    def __init__(self,t1_arr,mask_arr):
        '''
        输入数据的轴顺序为 x,y,z (nib.load 读取）
        :param t1_arr: numpy array, T1 序列数组
        :param mask_arr: numpy array, 脑部掩码数组
        '''
        self.t1_arr = t1_arr
        self.mask_arr = mask_arr


    def skull_detect(self):
        '''
        判断是否包含颅骨
        :return: result: BOOL, 为False表示不含颅骨，True表示含颅骨

        Usage:
        skull_de =  skull_detector(t1_arr,mask_arr)
        result = skull_de.skull_detect()
        '''
        max_layer_num,start_coor = self._find_max_layer_start_coor(self.mask_arr)
        n_arr,s_arr,w_arr,e_arr = self._get_edge_arr(layer = self.t1_arr[:,:,max_layer_num],start_coor = start_coor)
        n_arr = self._normalize_arr(n_arr)
        s_arr = self._normalize_arr(s_arr)
        w_arr = self._normalize_arr(w_arr)
        e_arr = self._normalize_arr(e_arr)

        result, arr_result = self._judge_4_arr(n_arr, s_arr, w_arr, e_arr, threshold=2)
        return result

    def _find_max_layer_start_coor(self,data):
        '''
        找到掩码面积最大的层，找到上下左右四个方向的起始坐标
        :param data: numpy array, 务必传入进行了颅骨分割之后的掩码
        :return:
        max_layer: int, 掩码区域最大的层编号
        start_coor: list of tuple, 进行边界数值查询的 起点坐标
        '''

        max_layer = 0
        num_pixel = 0
        temp = copy.deepcopy(data)
        temp[temp>0] = 1
        for i in range(data.shape[2]):
            if np.sum(temp[:,:,i]) > num_pixel:
                num_pixel = np.sum(temp[:,:,i])
                max_layer = i

        x_center = int(data.shape[0]/2)
        y_center = int(data.shape[1]/2)
        # N,S,W,E [(x,y),(x,y),(x,y),(x,y)]
        start_coor = []
        #N----- (x_center,n) ,从中心点出发，向上找到第一个掩码为零的坐标
        for i in range(y_center,0,-1):
            if temp[x_center,i,max_layer] == 0:
                start_coor.append((x_center,i+5))
                break
        #如果没有找到起始坐标，则直接安排坐标
        if len(start_coor) == 0:
            start_coor.append((x_center, 25))

        #S----  (x_center,s) , 从中心点出发，向下找到第一个掩码为零的坐标
        for i in range(y_center,data.shape[1]):
            if temp[x_center,i,max_layer] == 0:
                start_coor.append((x_center,i-5))
                break
        if len(start_coor) == 1:
            start_coor.append((x_center, data.shape[1]- 25))

        #W----  (w,y_center) , 从中心点出发，向左找到第一个掩码为零的坐标
        for i in range(x_center,0,-1):
            if temp[i,y_center,max_layer] == 0:
                start_coor.append((i+5,y_center))
                break
        if len(start_coor) == 2:
            start_coor.append((25, y_center))

        #E----- (e,y_center) , 从中心点出发，向右找到第一个掩码为零的坐标
        for i in range(x_center,data.shape[0]):
            if temp[i,y_center,max_layer] == 0:
                start_coor.append((i-5,y_center))
                break
        if len(start_coor) == 3:
            start_coor.append((data.shape[0]- 25, y_center))

        del temp
        return max_layer,start_coor

    def _get_edge_arr(self,layer,start_coor,arr_length = 40):
        '''
        找到上下左右四个方向上近似垂直于颅骨的法线数组
        :param layer:  numpy array, 脑部T1序列某一层的数据
        :param start_coor: list of tuple， 上下左右四个方向的起始坐标
        :param arr_length:  int， 法线数组长度
        :return: n_arr,s_arr,w_arr,e_arr： list of int， 四个方向的法线数组，反映了垂直于颅骨方向上的脑部数据的像素（强度、体素）变化
        '''
        # N ,x不变，y减小
        n_x,n_y = start_coor[0]
        n_arr = []
        while int(layer[n_x,n_y]) == 0:
            n_y += 1
        for i in range(n_y,0,-1):
            if len(n_arr) == arr_length:
                break
            n_arr.append(int(layer[n_x,i]))

        # S, x 不变，y增大
        s_x,s_y = start_coor[1]
        s_arr = []
        while int(layer[s_x,s_y]) == 0:
            s_y -= 1
        for i in range(s_y,layer.shape[1]):
            if len(s_arr) == arr_length:
                break
            s_arr.append(int(layer[s_x,i]))

        # W,  y不变， x减小
        w_x,w_y = start_coor[2]
        w_arr = []
        while int(layer[w_x,w_y]) == 0:
            w_x += 1
        for i in range(w_x,0 ,-1):
            if len(w_arr) == arr_length:
                break
            w_arr.append(int(layer[i,w_y]))

        # E ,y不变， x 减小
        e_x,e_y = start_coor[3]
        e_arr = []
        while int(layer[e_x,e_y]) == 0:
            e_x -= 1
        for i in range(e_x,layer.shape[0]):
            if len(e_arr) == arr_length:
                break
            e_arr.append(int(layer[i,e_y]))

        return n_arr,s_arr,w_arr,e_arr

    def _normalize_arr(self,arr):
        '''
        将法线数组数值 映射到 0-255
        :param arr:  list of int, 法线数组
        :return: temp_new: list of int, 映射后的法线数组
        '''
        temp = np.asarray(arr)
        min1= 0#np.min(temp)
        max1= np.max(temp)
        temp = ((temp-min1)/(max1-min1))*255
        temp = temp.tolist()
        temp_new = [int(x) for x in temp]
        return temp_new

    def _get_grad_arr(self,arr):
        '''
        计算法线数组各数值的变化率
        :param arr: list of int, 法线数组
        :return: grad_arr： list of float, 变化率数组，反映了法线数组下标由小到大的变化情况
        '''
        grad_arr = []
        for i in range(len(arr)-1):
            grad = (arr[i+1] - arr[i])/(arr[i]+1)
            grad_arr.append(round(grad,2))
        return grad_arr

    def _judge_grad(self,grad_arr, arr):
        '''
        对变化率数组进行判断，确认该变化率数组是否反映了有颅骨的脑部数据的变化情况
        :param grad_arr:  list of float， 变化率数组
        :param arr: list of int, 法线数组，便于做附加的大范围起伏判断
        :return:
        is_skull： int， 是否可能是颅骨边界， 1 代表是颅骨边界， 0 代表不是颅骨边界
        larger_than_1： int， 变化率数组中大于1的个数，（调试用）
        '''
        larger_than_1 = 0
        for i in range(len(grad_arr)):
            #记录变化率大于1 的情况
            if abs(grad_arr[i]) >= 1:
                larger_than_1 += 1
        is_skull = 0
        if larger_than_1 > 1:
            is_skull = 1
        else:
            #大范围起伏判断，取法线数组的前1/4的最小值，以及后3/4的最大值
            valley = np.min(arr[:int((len(arr))/4)])
            peak = np.max(arr[int((len(arr))/4):])
            large_scale_grad = (peak-valley)/(valley+1)
            if large_scale_grad > 1:
                is_skull = 1

        return is_skull, larger_than_1

    def _judge_4_arr(self,n_arr,s_arr,w_arr,e_arr, threshold = 2):
        '''
        对上下左右四个方向的颅骨边界判断情况进行汇总
        :param n_arr: list of int，上（北， NORTH）方向上的法线数组
        :param s_arr: list of int，下（南， SORTH）方向上的法线数组
        :param w_arr: list of int，左（西， WEST）方向上的法线数组
        :param e_arr: list of int，右（东， EAST）方向上的法线数组
        :param threshold:  int， 阈值，如果判断为有颅骨的方向个数，大于该阈值，则将该脑部图像判断为包含颅骨
        :return:
        is_skull：BOOL， 为False表示不含颅骨， 为True表示包含颅骨
        [n_is_skull,s_is_skull , w_is_skull , e_is_skull]： list of int, 上下左右各方向判断是否包含颅骨的情况，如果某个方向上的值为1，表示该方向判断为有颅骨
        '''
        n_arr_grad = self._get_grad_arr(n_arr)
        s_arr_grad = self._get_grad_arr(s_arr)
        w_arr_grad = self._get_grad_arr(w_arr)
        e_arr_grad = self._get_grad_arr(e_arr)

        n_is_skull, n_flag = self._judge_grad(n_arr_grad,n_arr)
        s_is_skull, s_flag = self._judge_grad(s_arr_grad,s_arr)
        w_is_skull, w_flag = self._judge_grad(w_arr_grad,w_arr)
        e_is_skull, e_flag = self._judge_grad(e_arr_grad,e_arr)

        is_skull = False
        vote_yes = n_is_skull + s_is_skull + w_is_skull + e_is_skull
        if vote_yes  > threshold:
            is_skull = True

        return is_skull,[n_is_skull,s_is_skull , w_is_skull , e_is_skull]
