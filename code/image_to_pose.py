import cv2
import mediapipe as mp

# 初始化姿态估计模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 读取图片
image = cv2.imread('test_image.jpg')

# 将图片转换为 RGB 格式（MediaPipe 需要 RGB 格式）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 使用姿态估计模型处理图片
results = pose.process(image_rgb)

# 打印关键点结果
if results.pose_landmarks:
    for index, landmark in enumerate(results.pose_landmarks.landmark):
        print(f'ID: {index}, X: {landmark.x:.2f}, Y: {landmark.y:.2f}, Z: {landmark.z:.2f}, visibility:{landmark.visibility:.2f}')
        # print(f'ID: {index}, visibility: {landmark.visibility:.2f}')
else:
    print("No pose landmarks detected.")

# 如果你想在图片上绘制关键点，可以使用以下代码
if results.pose_landmarks:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# 显示图片
cv2.imshow('MediaPipe Pose', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
