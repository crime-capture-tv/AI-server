from videoMAE_result import MultiVideoClassification
from preprocessing import Preprocessing
from classification_process import ClassificationProcess
import os
from typing import Union
from fastapi import FastAPI
import shutil
from datetime import datetime, timedelta

model_ckpt = "./models/20230916/checkpoint-599"

app = FastAPI()
preprocessing = Preprocessing()
video_classification = MultiVideoClassification(model_ckpt)
classification_process = ClassificationProcess()


@app.get('/')
def main():
    return 'classification api'


@app.get('/classification')
def preprocess(recordedAt: Union[str, None] = None, suspicionVideoPath: Union[str, None] = None):
    format_str = "%Y%m%d-%H-%M-%S"
    start_time="20230912-14-44-14"
    duration = 4
    counter_time = {
        'startTime': '20230915-21-57-40',
        'endTime': '20230915-21-57-59'
    }

    # use network path
    print(f'video downloading ... {suspicionVideoPath}')
    network_path = suspicionVideoPath
    network_file_name = network_path.split('\\')[-1]
    local_path = f'output/{network_file_name}'
    shutil.copy(network_path, local_path)
    print(f'video download complete!')

    input_file = local_path

    # Preprocessing 
    resolution = {'width': 640, 'height': 480}
    target_fps = 30

    preprocessing.file_load(input_file)
    preprocessing.regulate_resolution_fps(resolution, target_fps)
    preprocessing.segment_video(segment_duration=4, step_duration=2)
    preprocessing.crop_and_save_video()
    preprocessing.delete_short_videos(min_duration=2.0)

    # video classification
    input_file_path = '/'.join(input_file.split('/')[:-1])
    input_file_name = input_file.split('/')[-1].split('.')[0]
    video_path = os.path.join(input_file_path, f'{input_file_name}_crop')

    date=datetime.strptime(start_time,format_str)

    video_classification.load_videos(video_path)
    person_data = video_classification.predict(date, duration, format_str)
    # print(person_data)


    # result analyze
    status, catch_time = classification_process.check_behavior(person_data, counter_time, format_str)

    if status == 'Clear':
        print('Clear')

        resultText = {
            'result': 'All clear',
            'first_catch_time': catch_time[0][0],
            'last_catch_time': catch_time[-1][0],
            'suspicionVideoPath': suspicionVideoPath,
            'highlightVideoPath': None,
            'crimeType': 'normal'
        }

    elif status == 'Warning':
        print(f'Warning!  first_catch_time : {catch_time[0][0]}')

        # highlight file save
        try:
            catchvideopath = os.path.join(input_file_path, f'{input_file_name}_segment/segment_{catch_time[1][1]:03d}.mp4')
            print(catchvideopath)
            filename = f"hilight_{catch_time[1][0]}.mp4"
            hilight_network_path = fr'\\192.168.0.42\crimecapturetv\hilight-video\{filename}'
            print(f'Start saving highlight ...  {filename}')
            # shutil.copy(os.path.join(catchvideopath, catchvideo), network_path)
            with open(catchvideopath, 'rb') as src_file:
                with open(hilight_network_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
            print('Save complete')
        except TypeError as e:
            print('No file!')
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        resultText = {
            'result': 'Warning',
            'first_catch_time': catch_time[0][0],
            'last_catch_time': catch_time[-1][0],
            'suspicionVideoPath': suspicionVideoPath,
            'highlightVideoPath': hilight_network_path,
            'crimeType': 'theft'
        }



    print(resultText)
    return resultText