import copy

import numpy as np


# 动作计数器
class RepetitionCounter(object):
    # 计算给定目标姿势类的重复次数

    def __init__(self, flag, class_name, enter_threshold=8, exit_threshold=2, prev_result=None):
        self._flag = flag
        self._class_name = class_name

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
        self._shoulder_hip_height_ratio = 1.0
        self._shoulder_hip_height = 0.01
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
                        'shoulder_hip_height_ratio': self._shoulder_hip_height_ratio,
                        'wrist_distance_ratio': self._wrist_distance_ratio,
                        'hands_visib_left': self._hands_visib_left,
                        'hands_visib_right': self._hands_visib_right}

        if prev_result is not None:
            self._result = copy.deepcopy(prev_result)
            self._n_repeats = self._result['n_repeat']
            self._pose_entered = self._result['pose_entered']
            self._finished = self._result['finished']
            self._wrist_shoulder_angle_left = self._result['wrist_shoulder_angle_left']
            self._wrist_shoulder_angle_right = self._result['wrist_shoulder_angle_right']
            self._knee_angle = self._result['knee_angle']
            self._shoulder_hip_height_ratio = self._result['shoulder_hip_height_ratio']
            self._wrist_distance_ratio = self._result['wrist_distance_ratio']
            self._hands_visib_left = self._result['hands_visib_left']
            self._hands_visib_right = self._result['hands_visib_right']

    @property
    def n_repeats(self):
        return self._n_repeats

    def __call__(self, pose_classification, pose_landmarks):
        # 计算给定帧之前发生的重复次数
        # 我们使用两个阈值。首先，您需要从较高的位置上方进入姿势，然后您需要从较低的位置下方退出。
        # 阈值之间的差异使其对预测抖动稳定（如果只有一个阈值，则会导致错误计数）。

        # 参数：
        #   pose_classification：当前帧上的姿势分类字典
        #         Sample:
        #         {
        #             'squat_down': 8.3,
        #             'squat_up': 1.7,
        #         }

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
        if left_shoulder[3] > visibility_threshold and right_shoulder[3] > visibility_threshold:
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        elif left_shoulder[3] > visibility_threshold:
            shoulder_y = left_shoulder[1]
        elif right_shoulder[3] > visibility_threshold:
            shoulder_y = right_shoulder[1]
        if left_hip[3] > visibility_threshold and right_hip[3] > visibility_threshold:
            hip_y = (left_hip[1] + right_hip[1]) / 2
        elif left_shoulder[3] > visibility_threshold:
            hip_y = left_hip[1]
        elif right_shoulder[3] > visibility_threshold:
            hip_y = right_hip[1]
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
        if left_shoulder[3] > visibility_threshold and right_shoulder[3] > visibility_threshold and left_wrist[3] > visibility_threshold and right_wrist[3] > visibility_threshold:
            vector = left_shoulder[:3] - right_shoulder[:3]
            hip_distance = np.linalg.norm(vector)
            vector = left_wrist[:3] - right_wrist[:3]
            wrist_distance = np.linalg.norm(vector)
            wrist_distance_ratio = wrist_distance / hip_distance

        # 手的可见性
        hands_visib_left = pose_landmarks[16][3] + pose_landmarks[18][3] + pose_landmarks[20][3] + pose_landmarks[22][3]
        hands_visib_left /= 4
        hands_visib_right = pose_landmarks[15][3] + pose_landmarks[17][3] + pose_landmarks[19][3] + pose_landmarks[21][3]
        hands_visib_right /= 4
        # print("hands visibility:", hands_visib_left, hands_visib_right)

        # 获取姿势的置信度.
        pose_confidence = 0.0
        if self._class_name in pose_classification:
            pose_confidence = pose_classification[self._class_name]

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
                self._shoulder_hip_height = 0.01
                self._wrist_distance_ratio = 1.0
                self._hands_visib_left = 1.0
                self._hands_visib_right = 1.0
                self._finished = True

        if self._pose_entered:
            if self._shoulder_hip_height < shoulder_hip_height:
                self._shoulder_hip_height = shoulder_hip_height  # 设定基准背部高度，更新为捕捉到的最大值
            if self._wrist_shoulder_angle_left < wrist_shoulder_angle_left:
                self._wrist_shoulder_angle_left = wrist_shoulder_angle_left  # 更新为最大的角度
            if self._wrist_shoulder_angle_right < wrist_shoulder_angle_right:
                self._wrist_shoulder_angle_right = wrist_shoulder_angle_right  # 更新为最大的角度

        # 如果我们处于姿势并且正在退出它，则增加计数器并更新状态
        if self._pose_entered and pose_confidence < self._exit_threshold:

            if self._flag == 1:  # flag = 1 深蹲
                if "DeepSquat_up" in pose_classification and pose_classification["DeepSquat_up"] > self._enter_threshold:
                    self._n_repeats += 1
                    self._pose_entered = False
                    self._finished = False
                    self._result['n_repeat'] = self._n_repeats

            if self._flag == 2:  # flag = 2 提膝击掌
                if "HighKnees_left" in pose_classification and pose_classification["HighKnees_left"] > self._enter_threshold:
                    self._n_repeats += 1
                    self._pose_entered = False
                    self._finished = False

                if "HighKnees_right" in pose_classification and pose_classification["HighKnees_right"] > self._enter_threshold:
                    self._n_repeats += 1
                    self._pose_entered = False
                    self._finished = False

        if not self._finished and self._flag == 2:
            shoulder_hip_height_ratio = shoulder_hip_height / self._shoulder_hip_height
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

        if self._flag == 2:  # flag = 2 提膝击掌
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
            self._result['shoulder_hip_height_ratio'] = self._shoulder_hip_height_ratio
            # 指标4
            self._result['wrist_distance_ratio'] = self._wrist_distance_ratio
            self._result['hands_visib_left'] = self._hands_visib_left
            self._result['hands_visib_right'] = self._hands_visib_right

        return self._result
