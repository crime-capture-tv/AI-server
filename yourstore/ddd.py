import torch
import threading

def gpu_work(gpu_id):
    torch.cuda.set_device(gpu_id)
    tensor = torch.randn((1000, 1000), device='cuda')
    result = tensor @ tensor
    print(f"Result from GPU {gpu_id}: {result}")

# 사용 가능한 모든 GPU에 대해 스레드 생성
threads = []
for gpu_id in range(torch.cuda.device_count()):
    thread = threading.Thread(target=gpu_work, args=(gpu_id,))
    threads.append(thread)

# 스레드 시작
for thread in threads:
    thread.start()

# 모든 스레드가 완료될 때까지 대기
for thread in threads:
    thread.join()
