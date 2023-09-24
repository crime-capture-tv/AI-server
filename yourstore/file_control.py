import shutil
import os
import time

class FileControl():
    def __init__(self):
        pass

    def copy_with_progress(self, idx, src, dst, buffer_size=524288):  # 0.5MB = 524288 bytes            
        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
            total = fsrc.seek(0, 2)  # 파일 크기 확인
            fsrc.seek(0)  # 파일 포인터를 처음으로 돌림
            total_mb = total / (1024 * 1024)  # 전체 용량 MB
            
            copied = 0
            start_time = time.time()
            while True:
                buf = fsrc.read(buffer_size)
                if not buf:
                    break
                fdst.write(buf)
                copied += len(buf)
                
                elapsed_time = time.time() - start_time
                transfer_speed = copied / elapsed_time / (1024 * 1024)  # MB/s
                
                progress = (copied / total) * 100
                copied_mb = copied / (1024 * 1024)
                
                print(f"Copied {progress:.2f}%, Transferred {copied_mb:.2f}MB/{total_mb:.2f}MB at {transfer_speed:.2f}MB/s", end="\r")
            print()  # New line


    def download_file(self, network_path_1, network_path_2):

        # # 예제 사용
        # src_path = "path/to/source/file"
        # dst_path = "path/to/destination/file"

        # use network path
        network_file_name_1 = network_path_1.split('\\')[-1]
        local_path_1 = f'output/{network_file_name_1}'
        print(f'video downloading ... {network_file_name_1}')
        # shutil.copy(network_path_1, local_path_1)
        self.copy_with_progress(0, network_path_1, local_path_1)

        network_file_name_2 = network_path_2.split('\\')[-1]
        local_path_2 = f'output/{network_file_name_2}'
        print(f'video downloading ... {network_file_name_2}')
        # shutil.copy(network_path_2, local_path_2)
        self.copy_with_progress(0, network_path_2, local_path_2)
        print(f'video download complete!')

        return [local_path_1, local_path_2]
    

    def save_highlight(self, input_file, highlight_network_path, time, idex):
        input_file_path = '/'.join(input_file.split('/')[:-1])
        input_file_name = input_file.split('/')[-1].split('.')[0]
        try:
            catchvideopath = os.path.join(input_file_path, f'{input_file_name}_segment/segment_{idex:03d}.mp4')
            print(catchvideopath)
            filename = f"hilight_{time}.mp4"
            hight_network_full_path = os.path.join(highlight_network_path, filename)
            print(f'Start saving highlight ...  {filename}')
            with open(catchvideopath, 'rb') as src_file:
                with open(hight_network_full_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
            print('Save complete')
            return hight_network_full_path
        except TypeError as e:
            print('No file!')
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
