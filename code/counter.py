import os.path

import numpy as np


# 动作计数器
class RepetitionCounter(object):
    # 计算给定目标姿势类的重复次数

    def __init__(self, flag, enter_threshold=8, exit_threshold=2):

        if flag == 1:  # 深蹲
            self._flag = flag
            # 如果姿势通过了给定的阈值，那么我们就进入该动作的计数
            self._enter_threshold = enter_threshold
            self._exit_threshold = exit_threshold
            # 退出姿势的次数
            self._n_repeats = 0
            # 是否处于准备姿势
            self._pose_entered = False
            self._result = {'n_repeat': self._n_repeats}

        elif flag == 2:  # 提膝击掌
            self._flag = flag
            # 如果姿势通过了给定的阈值，那么我们就进入该动作的计数
            self._enter_threshold = enter_threshold
            self._exit_threshold = exit_threshold
            # 退出姿势的次数
            self._n_repeats = 0
            # 是否处于准备姿势
            self._pose_entered = False
            # 是否完成动作
            self._finished = True
            # 其他专家判断
            self._wrist_shoulder_angle_left = 0.0
            self._wrist_shoulder_angle_right = 0.0
            self._knee_angle = 0.0
            self._shoulder_hip_height_standard = 0.01
            self._shoulder_hip_height_ratio = 1.0
            self._wrist_distance_ratio = 1.0
            self._hands_visib_left = 1.0
            self._hands_visib_right = 1.0

            # 输出总结果
            self._result = {'n_repeat': self._n_repeats,
                            'pose_entered': self._pose_entered,
                            'finished': self._finished,
                            'wrist_shoulder_angle_left': self._wrist_shoulder_angle_left,
                            'wrist_shoulder_angle_right': self._wrist_shoulder_angle_right,
                            'knee_angle': self._knee_angle,
                            'shoulder_hip_height_standard': self._shoulder_hip_height_standard,
                            'shoulder_hip_height_ratio': self._shoulder_hip_height_ratio,
                            'wrist_distance_ratio': self._wrist_distance_ratio,
                            'hands_visib_left': self._hands_visib_left,
                            'hands_visib_right': self._hands_visib_right}

        elif flag == 3:  # 跳绳
            self._flag = flag
            # 退出姿势的次数
            self._n_repeats = 0
            # 是否处于准备姿势
            self._pose_highest_entered = False
            self._pose_lowest_entered = False
            # 其他专家判断
            self._hip_height_array = []
            self._window_len = 11
            self._hip_height_array2 = []
            self._window_len2 = 100
            self._shoulder_hip_height = 0
            self._hip_height_highest = 0
            self._hip_height_lowest = 0
            self._result = {'n_repeat': self._n_repeats,
                            'hip_height_array': self._hip_height_array,
                            'pose_highest_entered': self._pose_highest_entered,
                            'pose_lowest_entered': self._pose_lowest_entered}

    @property
    def n_repeats(self):
        return self._n_repeats

    def __call__(self, pose_classification, pose_landmarks):

        if self._flag == 1:  # 深蹲

            # 获取姿势的置信度.
            pose_confidence = 0.0
            if 'DeepSquat_down' in pose_classification:
                pose_confidence = pose_classification['DeepSquat_down']

            # On the very first frame or if we were out of the pose, just check if we
            # entered it on this frame and update the state.
            # 在第一帧或者如果我们不处于姿势中，只需检查我们是否在这一帧上进入该姿势并更新状态
            if not self._pose_entered:
                self._pose_entered = pose_confidence > self._enter_threshold

            # 如果我们处于姿势并且正在退出它，则增加计数器并更新状态
            if self._pose_entered and pose_confidence < self._exit_threshold:
                if "DeepSquat_up" in pose_classification and pose_classification["DeepSquat_up"] > self._enter_threshold:
                    self._n_repeats += 1
                    self._pose_entered = False
                    self._finished = False
                    self._result['n_repeat'] = self._n_repeats


        elif self._flag == 2:  # 提膝击掌

            # 肩膀高度和手腕高度
            left_shoulder = pose_landmarks[12]
            right_shoulder = pose_landmarks[11]
            left_wrist = pose_landmarks[16]
            right_wrist = pose_landmarks[15]
            visibility_threshold = 0.8  # 可见度阈值
            wrist_shoulder_angle_left = 0
            wrist_shoulder_angle_right = 0
            if left_shoulder[3] > visibility_threshold and left_wrist[3] > visibility_threshold:
                vector = left_wrist[:3] - left_shoulder[:3]
                cos_angle = np.dot(vector, np.array([0, 1, 0])) / (np.linalg.norm(vector))
                # 使用arccos计算角度，并将结果转换为度
                angle_rad = np.arccos(cos_angle)
                wrist_shoulder_angle_left = np.degrees(angle_rad)
            if right_shoulder[3] > visibility_threshold and right_wrist[3] > visibility_threshold:
                vector = right_wrist[:3] - right_shoulder[:3]
                cos_angle = np.dot(vector, np.array([0, 1, 0])) / (np.linalg.norm(vector))
                # 使用arccos计算角度，并将结果转换为度
                angle_rad = np.arccos(cos_angle)
                wrist_shoulder_angle_right = np.degrees(angle_rad)

            # 肩膀到胯部的距离（高度方向）
            left_hip = pose_landmarks[24]
            right_hip = pose_landmarks[23]
            hip_y = 0
            shoulder_y = 0
            shoulder_hip_height = 0.01
            if left_shoulder[3] > visibility_threshold and right_shoulder[3] > visibility_threshold:
                shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            elif left_shoulder[3] > visibility_threshold:
                shoulder_y = left_shoulder[1]
            elif right_shoulder[3] > visibility_threshold:
                shoulder_y = right_shoulder[1]
            if left_hip[3] > visibility_threshold and right_hip[3] > visibility_threshold:
                hip_y = (left_hip[1] + right_hip[1]) / 2
            elif left_hip[3] > visibility_threshold:
                hip_y = left_hip[1]
            elif right_hip[3] > visibility_threshold:
                hip_y = right_hip[1]
            if hip_y != 0 and shoulder_y != 0:
                shoulder_hip_height = hip_y - shoulder_y

            # 提膝角度
            left_knee = pose_landmarks[26]
            right_knee = pose_landmarks[25]
            knee_angle_left = 0
            knee_angle_right = 0
            if left_hip[3] > visibility_threshold and left_knee[3] > visibility_threshold:
                vector = left_knee[:3] - left_hip[:3]
                cos_angle = np.dot(vector, np.array([0, 1, 0])) / (np.linalg.norm(vector))
                # 使用arccos计算角度，并将结果转换为度
                angle_rad = np.arccos(cos_angle)
                knee_angle_left = np.degrees(angle_rad)
            if right_hip[3] > visibility_threshold and right_knee[3] > visibility_threshold:
                vector = right_knee[:3] - right_hip[:3]
                cos_angle = np.dot(vector, np.array([0, 1, 0])) / (np.linalg.norm(vector))
                # 使用arccos计算角度，并将结果转换为度
                angle_rad = np.arccos(cos_angle)
                knee_angle_right = np.degrees(angle_rad)
            knee_angle = max(knee_angle_left, knee_angle_right)

            # 两手最近距离（与双肩距离的比例）
            wrist_distance_ratio = 1
            if left_shoulder[3] > visibility_threshold and right_shoulder[3] > visibility_threshold and left_wrist[
                3] > visibility_threshold and right_wrist[3] > visibility_threshold:
                vector = left_shoulder[:3] - right_shoulder[:3]
                hip_distance = np.linalg.norm(vector)
                vector = left_wrist[:3] - right_wrist[:3]
                wrist_distance = np.linalg.norm(vector)
                wrist_distance_ratio = wrist_distance / hip_distance

            # 手的可见性
            hands_visib_left = pose_landmarks[16][3] + pose_landmarks[18][3] + pose_landmarks[20][3] + \
                               pose_landmarks[22][3]
            hands_visib_left /= 4
            hands_visib_right = pose_landmarks[15][3] + pose_landmarks[17][3] + pose_landmarks[19][3] + \
                                pose_landmarks[21][3]
            hands_visib_right /= 4
            # print("hands visibility:", hands_visib_left, hands_visib_right)

            # 获取姿势的置信度.
            pose_confidence = 0.0
            if 'HighKnees_prepare' in pose_classification:
                pose_confidence = pose_classification['HighKnees_prepare']

            # On the very first frame or if we were out of the pose, just check if we
            # entered it on this frame and update the state.
            # 在第一帧或者如果我们不处于姿势中，只需检查我们是否在这一帧上进入该姿势并更新状态
            if not self._pose_entered:
                self._pose_entered = pose_confidence > self._enter_threshold
                if self._pose_entered:  # 初始化
                    self._wrist_shoulder_angle_left = 0.0
                    self._wrist_shoulder_angle_right = 0.0
                    self._knee_angle = 0.0
                    self._shoulder_hip_height_ratio = 1.0
                    self._shoulder_hip_height_standard = 0.01
                    self._wrist_distance_ratio = 1.0
                    self._hands_visib_left = 1.0
                    self._hands_visib_right = 1.0
                    self._finished = True

            if self._pose_entered:
                if self._shoulder_hip_height_standard < shoulder_hip_height:
                    self._shoulder_hip_height_standard = shoulder_hip_height  # 设定基准背部高度，更新为捕捉到的最大值
                if self._wrist_shoulder_angle_left < wrist_shoulder_angle_left:
                    self._wrist_shoulder_angle_left = wrist_shoulder_angle_left  # 更新为最大的角度
                if self._wrist_shoulder_angle_right < wrist_shoulder_angle_right:
                    self._wrist_shoulder_angle_right = wrist_shoulder_angle_right  # 更新为最大的角度

            # 如果我们处于姿势并且正在退出它，则增加计数器并更新状态
            if self._pose_entered and pose_confidence < self._exit_threshold:

                if "HighKnees_left" in pose_classification and pose_classification[
                    "HighKnees_left"] > self._enter_threshold:
                    self._n_repeats += 1
                    self._pose_entered = False
                    self._finished = False

                if "HighKnees_right" in pose_classification and pose_classification[
                    "HighKnees_right"] > self._enter_threshold:
                    self._n_repeats += 1
                    self._pose_entered = False
                    self._finished = False

            if not self._finished:
                shoulder_hip_height_ratio = shoulder_hip_height / self._shoulder_hip_height_standard
                if shoulder_hip_height_ratio < self._shoulder_hip_height_ratio:
                    self._shoulder_hip_height_ratio = shoulder_hip_height_ratio
                if knee_angle > self._knee_angle:
                    self._knee_angle = knee_angle
                if wrist_distance_ratio < self._wrist_distance_ratio:
                    self._wrist_distance_ratio = wrist_distance_ratio
                if hands_visib_left < self._hands_visib_left:
                    self._hands_visib_left = hands_visib_left
                if hands_visib_right < self._hands_visib_right:
                    self._hands_visib_right = hands_visib_right

            # 更新结果
            self._result['n_repeat'] = self._n_repeats
            self._result['pose_entered'] = self._pose_entered
            self._result['finished'] = self._finished
            # 指标1
            self._result['wrist_shoulder_angle_left'] = self._wrist_shoulder_angle_left
            self._result['wrist_shoulder_angle_right'] = self._wrist_shoulder_angle_right
            # 指标2
            self._result['knee_angle'] = self._knee_angle
            # 指标3
            self._result['shoulder_hip_height_standard'] = self._shoulder_hip_height_standard
            self._result['shoulder_hip_height_ratio'] = self._shoulder_hip_height_ratio
            # 指标4
            self._result['wrist_distance_ratio'] = self._wrist_distance_ratio
            self._result['hands_visib_left'] = self._hands_visib_left
            self._result['hands_visib_right'] = self._hands_visib_right

            # csv_file_path = 'results.csv'
            # # 使用'w'模式打开CSV文件，如果文件已存在则会被覆盖
            # # 如果你想在现有文件上追加数据，可以使用'a'模式
            # with open(csv_file_path, mode='a', newline='') as csv_file:
            #     writer = csv.writer(csv_file)
            #     writer.writerow([self._n_repeats,
            #                      self._pose_entered,
            #                      self._finished,
            #                      wrist_shoulder_angle_left,
            #                      self._wrist_shoulder_angle_left,
            #                      wrist_shoulder_angle_right,
            #                      self._wrist_shoulder_angle_right,
            #                      knee_angle_left,
            #                      knee_angle_right,
            #                      self._knee_angle,
            #                      shoulder_hip_height,
            #                      self._shoulder_hip_height_standard,
            #                      self._shoulder_hip_height_ratio,
            #                      self._wrist_distance_ratio,
            #                      hands_visib_left,
            #                      self._hands_visib_left,
            #                      hands_visib_right,
            #                      self._hands_visib_right])

        elif self._flag == 3:  # 跳绳

            # 胯部的高度
            left_hip = pose_landmarks[24]
            right_hip = pose_landmarks[23]
            hip_y = 0
            visibility_threshold = 0.8  # 可见度阈值
            if left_hip[3] > visibility_threshold and right_hip[3] > visibility_threshold:
                hip_y = (left_hip[1] + right_hip[1]) / 2
            elif left_hip[3] > visibility_threshold:
                hip_y = left_hip[1]
            elif right_hip[3] > visibility_threshold:
                hip_y = right_hip[1]
            if hip_y != 0:
                self._hip_height_array.append(hip_y)
            if len(self._hip_height_array) > self._window_len:
                self._hip_height_array.pop(0)
            if hip_y != 0:
                self._hip_height_array2.append(hip_y)
            if len(self._hip_height_array2) > self._window_len2:
                self._hip_height_array2.pop(0)

            # 肩膀高度
            shoulder_y = 0
            left_shoulder = pose_landmarks[12]
            right_shoulder = pose_landmarks[11]
            if left_shoulder[3] > visibility_threshold and right_shoulder[3] > visibility_threshold:
                shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            elif left_shoulder[3] > visibility_threshold:
                shoulder_y = left_shoulder[1]
            elif right_shoulder[3] > visibility_threshold:
                shoulder_y = right_shoulder[1]
            if hip_y != 0 and shoulder_y != 0:
                self._shoulder_hip_height = hip_y - shoulder_y

            mid_index = int((self._window_len - 1) / 2)
            arr = np.array(self._hip_height_array)
            # 找到最大值的索引
            max_index = np.argmax(arr)
            # 找到最小值的索引
            min_index = np.argmin(arr)
            if min_index == mid_index:
                self._pose_highest_entered = True
                self._pose_lowest_entered = False
                self._hip_height_highest = self._hip_height_array[mid_index]
                if (self._hip_height_lowest - self._hip_height_highest > 1 / 8 * self._shoulder_hip_height):
                    self._n_repeats += 1
                    # import matplotlib.pyplot as plt
                    # plt.scatter(np.arange(len(self._hip_height_array2)), (-1) * np.array(self._hip_height_array2))
                    # import time
                    # current_time = time.time() * 1000
                    # plt.savefig(os.path.join('scatter', f'{current_time}.png'))
                    # plt.clf()
                self._highest_to_lowest = 0
                self._lowest_to_highest = 0
            if max_index == mid_index:
                self._pose_lowest_entered = True
                self._pose_highest_entered = False
                self._hip_height_lowest = self._hip_height_array[mid_index]

            self._result['n_repeat'] = self._n_repeats
            self._result['hip_height_array'] = self._hip_height_array
            self._result['pose_highest_entered'] = self._pose_highest_entered
            self._result['pose_lowest_entered'] = self._pose_lowest_entered

        return self._result
