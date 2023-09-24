from threading import Thread
from queue import Queue
from datetime import datetime
import os

from videoMAE_result import MultiVideoClassification  # 여러분의 모듈 경로를 'your_module'로 변경하세요
from preprocessing import Preprocessing
from file_control import DownloadNetworkFile

def worker(thread_id, input_file, date, format_str, output_queue):
    classifier = MultiVideoClassification(model_ckpt)
    # Preprocessing 
    preprocessing = Preprocessing()
    resolution = {'width': 640, 'height': 480}
    target_fps = 30

    duration = 4

    preprocessing.file_load(input_file)
    preprocessing.regulate_resolution_fps(resolution, target_fps)
    preprocessing.segment_video(segment_duration=4, step_duration=2)
    preprocessing.crop_and_save_video()
    preprocessing.delete_short_videos(min_duration=2.0)

    # Load preprocessed files

    classifier.load_videos(input_file)
    result_dic = classifier.predict(date, duration, format_str)
    output_queue.put((thread_id, result_dic))



if __name__ == '__main__':
    import time
    start = time.time()

    output_queue = Queue()
    # download_files = DownloadNetworkFile()

    # 스레드에 전달할 인수를 설정합니다
    model_ckpt = "./models/20230914/checkpoint-426"

    input_file_1 = 'data/multi_cam/insert_1.mp4'
    input_file_2 = 'data/multi_cam/insert_2.mp4'

    # video_dir1 = 'data/multi_cam/insert_1_crop'
    # video_dir2 = 'data/multi_cam/insert_2_crop'
    date = datetime.now()
    
    format_str = "%Y-%m-%d %H:%M:%S"  # 수정할 필요가 있을 수 있습니다

    # classifier = MultiVideoClassification(model_ckpt)

    # 두 개의 스레드를 생성하고 시작합니다
    thread1 = Thread(target=worker, args=(1, input_file_1, date, format_str, output_queue))
    thread2 = Thread(target=worker, args=(2, input_file_2, date, format_str, output_queue))

    thread1.start()
    thread2.start()

    # 모든 스레드가 종료될 때까지 기다립니다
    thread1.join()
    thread2.join()

    # 결과를 저장할 변수를 초기화합니다
    result_dic_1 = None
    result_dic_2 = None

    # 결과를 출력합니다
    while not output_queue.empty():
        thread_id, result_dic = output_queue.get()
        if thread_id == 1:
            result_dic_1 = result_dic
        elif thread_id == 2:
            result_dic_2 = result_dic
    
    # 각 스레드의 결과를 개별적으로 출력합니다
    print("Thread 1 results:")
    print(result_dic_1)
    
    print("Thread 2 results:")
    print(result_dic_2)

    result_dic = {}
    for key1, key2 in zip(result_dic_1, result_dic_2):
        # print(f'{key1}: {result_dic_1[key1]}, {key2}: {result_dic_2[key2]}')
        # print(f'{key1}: {result_dic_1[key1][1]}')
        if result_dic_1[key1][1] == result_dic_2[key2][1]:
            # print(result_dic_1[key1][1])
            result_dic[key1] = result_dic_1[key1]
        else:
            # print('pass')
            result_dic[key1] = [result_dic_1[key1][0], 'pass']

    # print(result_dic)

    end = time.time()
    print(f"run time :  {end - start:.5f} sec")