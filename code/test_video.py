import videoprocess as vp
import trainingsetprocess as tp

if __name__ == '__main__':
    # flag = 1 深蹲
    # flag = 2 提膝击掌
    # flag = 3 跳绳
    # flag = 4 俯卧撑
    # flag = 5 仰卧起坐
    flag = 5
    # 输入要处理的视频的路径
    video_path = r".\video-sample\仰卧起坐\SitUp19.mp4"
    # tp.trainset_process(flag)  # 训练新的动作姿态图片
    vp.video_process(video_path, flag)
