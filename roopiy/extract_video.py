import subprocess
import os
import docpie


def detect_fps(target_path: str) -> float:
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]
    output = subprocess.check_output(command).decode().strip().split('/')
    # try:
    #     numerator, denominator = map(int, output)
    #     return numerator / denominator
    # except Exception:
    #     pass
    # return 30
    numerator, denominator = map(int, output)
    return numerator / denominator


def extract_frames(target_path: str, temp_directory_path: str, temp_frame_quality: int = 0, fps: float = 30) -> bool:
    temp_frame_quality = temp_frame_quality * 31 // 100
    subprocess.check_output(
        ['ffmpeg',
         '-hide_banner',
         '-loglevel', 'error',
         '-hwaccel', 'auto',
         '-i', target_path,
         '-q:v', str(temp_frame_quality),
         '-pix_fmt', 'rgb24',
         '-vf', 'fps=' + str(fps),
         os.path.join(temp_directory_path, '%06d.png')
         ]
    )


def by_args(args: dict[str, str]):
    video_file: str = args['<video_file>']
    output_dir: str = args['<frames_dir>']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fps: int = int(args['--fps'])
    actual_fps: float = fps
    if fps == -1:
        actual_fps = detect_fps(video_file)

    extract_frames(video_file, output_dir, fps=actual_fps)
