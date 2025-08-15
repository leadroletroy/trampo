import subprocess
import time
import psutil
import threading
import os
import signal

try:
    import GPUtil
    use_gpu_monitor = True
except ImportError:
    use_gpu_monitor = False
    print("Install GPUtil with `pip install gputil` for GPU monitoring.")

# Customizable thresholds (adjust to your needs)
MAX_CPU = 95.0       # in %
MAX_RAM = 85.0       # in %
MAX_GPU_MEM_UTIL = 0.90  # GPU VRAM usage (0.90 = 90%)

cameras = ['Camera1_M11139', 'Camera2_M11140', 'Camera3_M11141', 'Camera4_M11458', 'Camera5_M11459', 'Camera6_M11461', 'Camera7_M11462', 'Camera8_M11463']

#cameras = ['Camera4_M11458', 'Camera5_M11459', 'camera6_M11461', 'Camera7_M11462', 'Camera8_M11463']

path = '/mnt/D494C4CF94C4B4F0/Trampoline_avril2025/Videos_trampo_avril2025/20250429'
routines = ['1_partie_0429_000-', '1_partie_0429_001-', '1_partie_0429_002-', '1_partie_0429_003-',
            '1_partie_0429_004-', '1_partie_0429_005-', '1_partie_0429_006-', '1_partie_0429_007-',
            '1_partie_0429_008-', '1_partie_0429_009-', '1_partie_0429_010-', '1_partie_0429_011-',
            '1_partie_0429_013-', '1_partie_0429_014-', '1_partie_0429_015-', '1_partie_0429_016-',
            '1_partie_0429_017-', '1_partie_0429_018-', '1_partie_0429_019-', '1_partie_0429_020-',
            '1_partie_0429_021-', '1_partie_0429_022-', '1_partie_0429_023-', '1_partie_0429_024-',
            '1_partie_0429_025-', '1_partie_0429_026-', '1_partie_0429_027-']

video_sources = [path+'/'+routine+cam+'.mp4' for routine in routines for cam in cameras]

def monitor_and_kill(proc, stop_event):
    while not stop_event.is_set() and proc.poll() is None:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        kill_reason = None

        if cpu > MAX_CPU:
            kill_reason = f"CPU usage {cpu:.1f}% > {MAX_CPU}%"
        elif ram > MAX_RAM:
            kill_reason = f"RAM usage {ram:.1f}% > {MAX_RAM}%"

        if use_gpu_monitor:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.memoryUtil > MAX_GPU_MEM_UTIL:
                    kill_reason = f"GPU VRAM {gpu.memoryUtil*100:.1f}% > {MAX_GPU_MEM_UTIL*100:.1f}%"
                    break

        if kill_reason:
            print(f"\n[🔥 RESOURCE LIMIT EXCEEDED] {kill_reason}")
            print("[❌ Killing process to prevent crash...]")

            try:
                proc.send_signal(signal.SIGINT)
                time.sleep(3)
                if proc.poll() is None:
                    proc.kill()
            except Exception as e:
                print(f"[⚠️ ERROR] Could not kill process: {e}")
            stop_event.set()
            return

        time.sleep(10)


def run_tracking(video_path):
    cmd = [
        "python", "4D-Humans/track.py",
        f"video.source={video_path}",
        "render.type=HUMAN_MESH",
        "render.show_keypoints=True",
        "device='cuda'"
    ]
    print(f"\n>>> Running: {' '.join(cmd)}\n")
    process = subprocess.Popen(cmd)
    return process

def main():
    for video_path in video_sources:
        output_path = 'demo_' + video_path.split('/')[-1].split('.')[0]+'.pkl'
        if output_path not in os.listdir('/home/lea/trampo/4DHumans/outputs/results_focal'):
            stop_event = threading.Event()
            process = run_tracking(video_path)

            monitor_thread = threading.Thread(target=monitor_and_kill, args=(process, stop_event))
            monitor_thread.start()

            process.wait()
            stop_event.set()
            monitor_thread.join()

            print(f"\n✅ Finished or killed: {os.path.basename(video_path)}\n")
            time.sleep(10)
        else:
            print(f'SKIPPING {output_path}, already exists.')
        
        break

if __name__ == "__main__":
    main()
