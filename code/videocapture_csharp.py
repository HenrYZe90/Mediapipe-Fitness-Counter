import datetime
import os
import cv2
import numpy as np
import poseembedding as pe  # 姿态关键点编码模块
import poseclassifier as pc  # 姿态分类器
import resultsmooth as rs  # 分类结果平滑
import counter_csharp  # 动作计数器


def process(flag, pose_landmarks, counter_result):
    # class_name需要与你的训练样本的两个动作状态图像文件夹的名字中的一个(或者是与fitness_poses_csvs_out中的一个csv文件的名字）保持一致，它后面将用于分类时的索引。
    # 具体是哪个动作文件夹的名字取决于你的运动是什么，例如：如果是深蹲，明显比较重要的判断计数动作是蹲下去；如果是引体向上，则判断计数的动作是向上拉到最高点的那个动作
    # class_name = 'squat_down'
    # out_video_path = 'squat-sample-out.mp4'
    if flag == 1:
        class_name = 'DeepSquat_down'
    elif flag == 2:
        class_name = 'HighKnees_prepare'

    # Initialize tracker, classifier and counter.
    # Do that before every video as all of them have state.

    # Folder with pose class CSVs. That should be the same folder you using while building classifier to output CSVs.
    pose_samples_folder = 'fitness_poses_csvs_out'

    # Initialize embedder.
    # 得到表示肢体动作的向量
    pose_embedder = pe.FullBodyPoseEmbedder()

    # Initialize classifier.
    # Check that you are using the same parameters as during bootstrapping.
    # 动作分类 动作A:2, 动作B:8 ...
    pose_classifier = pc.PoseClassifier(
        pose_samples_folder=pose_samples_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # Initialize counter.
    # 计数+判断动作标准
    repetition_counter = counter_csharp.RepetitionCounter(
        flag=flag,
        class_name=class_name,
        prev_result=counter_result)

    if pose_landmarks is not None:
        # Get landmarks.
        assert pose_landmarks.shape == (33, 4), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

        # Classify the pose on the current frame.
        pose_classification = pose_classifier(pose_landmarks[:, :3])
        print('完成时间 ', datetime.datetime.now())
        print('动作分类 ', pose_classification)

        # Count repetitions.
        counter_result = repetition_counter(pose_classification, pose_landmarks)
        print('输出结果 ', counter_result)

