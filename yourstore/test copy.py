from threading import Thread
from queue import Queue
from datetime import datetime
import os

from videoMAE_result import MultiVideoClassification  # 여러분의 모듈 경로를 'your_module'로 변경하세요
from preprocessing import Preprocessing
# from file_control import DownloadNetworkFile

model_ckpt = "./models/20230916/checkpoint-599"

preprocessing = Preprocessing()
video_classification = MultiVideoClassification(model_ckpt)
# download_files = DownloadNetworkFile()


if __name__ == '__main__':
    import time
    start = time.time()

    format_str = "%Y%m%d-%H-%M-%S"
    start_time="20230912-14-44-14"
    duration = 4
    counter_time = {
        'startTime': '20230915-21-57-40',
        'endTime': '20230915-21-57-59'
    }

    input_file_1 = 'data/multi_cam/insert_1.mp4'
    input_file_2 = 'data/multi_cam/insert_2.mp4'

    date = datetime.now()

    # Preprocessing 
    resolution = {'width': 640, 'height': 480}
    target_fps = 30

    input_files = [input_file_1, input_file_2]
    for input_file in input_files:
        preprocessing.file_load(input_file)
        preprocessing.regulate_resolution_fps(resolution, target_fps)
        preprocessing.segment_video(segment_duration=4, step_duration=2)
        preprocessing.crop_and_save_video()
        preprocessing.delete_short_videos(min_duration=2.0)

        # Load preprocessed files

        video_classification.load_videos(input_file)
        result_dic = video_classification.predict(date, duration, format_str)
    
    end = time.time()
    print(f"run time :  {end - start:.5f} sec")