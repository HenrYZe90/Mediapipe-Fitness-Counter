# 动作计数器
class RepetitionCounter(object):
    # 计算给定目标姿势类的重复次数

    def __init__(self, flag, class_name, enter_threshold=8, exit_threshold=2):
        self._flag = flag
        self._class_name = class_name

        # 如果姿势通过了给定的阈值，那么我们就进入该动作的计数
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # 是否处于给定的姿势
        self._pose_entered = False

        # 退出姿势的次数
        self._n_repeats = 0

        # 其他专家判断
        self._if_wrist_over_shoulder = True
        self._if_straighten_back = True
        self._shoulder_hip = 0

        # 输出总结果
        self._result = {'n_repeat': 0, 'if_wrist_over_shoulder': True, 'if_straighten_back': True}

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
        wrist_over_shoulder = 0  # 手腕高度 - 肩膀高度
        visibility_threshold = 0.6  # 可见度阈值
        if left_shoulder.visibility > visibility_threshold and left_wrist.visibility > visibility_threshold:
            wrist_over_shoulder += left_shoulder.y - left_wrist.y
        if right_shoulder.visibility > visibility_threshold and right_wrist.visibility > visibility_threshold:
            wrist_over_shoulder += right_shoulder.y - right_wrist.y

        # 肩膀到胯部的距离（高度方向）
        left_hip = pose_landmarks[23]
        right_hip = pose_landmarks[24]
        if left_shoulder.visibility > visibility_threshold and right_shoulder.visibility > visibility_threshold:
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        elif left_shoulder.visibility > visibility_threshold:
            shoulder_y = left_shoulder.y
        elif right_shoulder.visibility > visibility_threshold:
            shoulder_y = right_shoulder.y
        if left_hip.visibility > visibility_threshold and right_hip.visibility > visibility_threshold:
            hip_y = (left_hip.y + right_hip.y) / 2
        elif left_shoulder.visibility > visibility_threshold:
            hip_y = left_hip.y
        elif right_shoulder.visibility > visibility_threshold:
            hip_y = right_hip.y
        shoulder_hip = hip_y - shoulder_y

        # 获取姿势的置信度.
        pose_confidence = 0.0
        if self._class_name in pose_classification:
            pose_confidence = pose_classification[self._class_name]

        # On the very first frame or if we were out of the pose, just check if we
        # entered it on this frame and update the state.
        # 在第一帧或者如果我们不处于姿势中，只需检查我们是否在这一帧上进入该姿势并更新状态
        if not self._pose_entered:
            self._pose_entered = pose_confidence > self._enter_threshold
            if self._pose_entered:
                self._shoulder_hip = shoulder_hip  # 设定基准背部高度
                self._if_straighten_back = True
                self._result['if_straighten_back'] = self._if_straighten_back
                if wrist_over_shoulder > 0:
                    self._if_wrist_over_shoulder = True
                else:
                    self._if_wrist_over_shoulder = False
            self._result['if_wrist_over_shoulder'] = self._if_wrist_over_shoulder
            return self._result

        if self._pose_entered and not self._if_wrist_over_shoulder and wrist_over_shoulder > 0:
            self._if_wrist_over_shoulder = True
            self._result['if_wrist_over_shoulder'] = self._if_wrist_over_shoulder

        # 如果我们处于姿势并且正在退出它，则增加计数器并更新状态
        if self._pose_entered and pose_confidence < self._exit_threshold:
            self._if_wrist_over_shoulder = True
            self._result['if_wrist_over_shoulder'] = self._if_wrist_over_shoulder

            if self._flag == 1:  # flag = 1 深蹲
                if "DeepSquat_up" in pose_classification and pose_classification["DeepSquat_up"] > self._enter_threshold:
                    self._n_repeats += 1
                    self._pose_entered = False
                    self._result['n_repeat'] = self._n_repeats
                    return self._result

            if self._flag == 2:  # flag = 2 提膝击掌
                if "HighKnees_left" in pose_classification and pose_classification["HighKnees_left"] > self._enter_threshold:
                    self._n_repeats += 1
                    self._pose_entered = False
                    self._if_wrist_over_shoulder = True
                    self._result['n_repeat'] = self._n_repeats
                    self._result['if_wrist_over_shoulder'] = self._if_wrist_over_shoulder
                    if shoulder_hip < 0.85 * self._shoulder_hip:
                        self._if_straighten_back = False
                        self._result['if_straighten_back'] = self._if_straighten_back
                    return self._result
                if "HighKnees_right" in pose_classification and pose_classification["HighKnees_right"] > self._enter_threshold:
                    self._n_repeats += 1
                    self._pose_entered = False
                    self._if_wrist_over_shoulder = True
                    self._result['n_repeat'] = self._n_repeats
                    self._result['if_wrist_over_shoulder'] = self._if_wrist_over_shoulder
                    if shoulder_hip < 0.85 * self._shoulder_hip:
                        self._if_straighten_back = False
                        self._result['if_straighten_back'] = self._if_straighten_back
                    return self._result

        return self._result
