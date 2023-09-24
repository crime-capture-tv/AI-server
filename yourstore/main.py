from typing import Union, Optional
from fastapi import FastAPI
from datetime import datetime
from pydantic import BaseModel

from videoMAE_result import MultiVideoClassification
from preprocessing import Preprocessing
from classification_process import ClassificationProcess
from file_control import FileControl

model_ckpt = "./models/20230916/checkpoint-599"

app = FastAPI()
preprocessing = Preprocessing()
video_classification = MultiVideoClassification(model_ckpt)
classification_process = ClassificationProcess()
file_control = FileControl()

class Item(BaseModel):
    suspicionVideoPath01: str
    suspicionVideoPath02: str
    stayStartTime: Optional[str] = None
    stayEndTime: Optional[str] = None


@app.get('/')
def main():
    return 'classification api'


@app.post('/classification')
# def preprocess(suspicionVideoPath01: Union[str, None] = None, suspicionVideoPath02: Union[str, None] = None):
def preprocess(item: Item):
    print(item)
    format_str = "%Y-%m-%d-%H-%M-%S"
    start_time="2023-09-12-14-44-14"
    duration = 4
    counter_time = {
        'startTime': item.stayStartTime,
        'endTime': item.stayEndTime 
    }

    # use network path
    local_paths = file_control.download_file(item.suspicionVideoPath01, item.suspicionVideoPath02)

    # test local video
    # local_paths = ['data/multi_cam/insert_1.mp4', 'data/multi_cam/insert_2.mp4']

    resolution = {'width': 640, 'height': 480}
    target_fps = 30

    date=datetime.strptime(start_time,format_str)

    input_files = local_paths
    for idx, input_file in enumerate(input_files):
        # Preprocessing 
        preprocessing.file_load(input_file)
        preprocessing.regulate_resolution_fps(resolution, target_fps)
        preprocessing.segment_video(segment_duration=4, step_duration=2)
        preprocessing.crop_and_save_video()
        preprocessing.delete_short_videos(min_duration=2.0)

        # Video classification 
        video_classification.load_videos(input_file)
        person_data = video_classification.predict(date, duration, format_str)
        if idx == 0:
            result_dic_1 = person_data
        elif idx == 1:
            result_dic_2 = person_data

    # print(result_dic)

    # result analyze
    classification_process.compare_results(result_dic_1, result_dic_2)
    status, catch_time = classification_process.check_behavior(counter_time, format_str)
    print(catch_time)

    if status == 'Clear':
        print('result : Clear')

        resultText = {
            'result': 'All clear',
            # 'first_catch_time': catch_time[0][0],
            # 'last_catch_time': catch_time[-1][0],
            # 'suspicionVideoPath': suspicionVideoPath,
            'highlightVideoPath': 'no highlight',
            'crimeType': 'normal'
        }

    elif status == 'Warning':
        print(f'Warning!  first_catch_time : {catch_time[0][0]}')

        # highlight file save
        # try:
        #     catchvideopath = os.path.join(input_file_path, f'{input_file_name}_segment/segment_{catch_time[1][1]:03d}.mp4')
        #     print(catchvideopath)
        #     filename = f"hilight_{catch_time[1][0]}.mp4"
        #     hightlight_network_path = fr'\\192.168.0.42\crimecapturetv\hilight-video\{filename}'
        #     print(f'Start saving highlight ...  {filename}')
        #     # shutil.copy(os.path.join(catchvideopath, catchvideo), network_path)
        #     with open(catchvideopath, 'rb') as src_file:
        #         with open(hightlight_network_path, 'wb') as dst_file:
        #             dst_file.write(src_file.read())
        #     print('Save complete')
        # except TypeError as e:
        #     print('No file!')
        # except Exception as e:
        #     print(f"An unexpected error occurred: {e}")
        
        hightlight_network_path = fr'\\192.168.0.26\crimecapturetv\highlight-video'
        highlight_network_full_path = file_control.save_highlight(input_file, hightlight_network_path, catch_time[1][0], catch_time[1][1])

        resultText = {  
            'result': 'Warning',
            # 'first_catch_time': catch_time[0][0],
            # 'last_catch_time': catch_time[-1][0],
            # 'suspicionVideoPath': suspicionVideoPath,
            'highlightVideoPath': highlight_network_full_path,
            'crimeType': 'theft'
        }


    
    print(resultText)
    return resultText