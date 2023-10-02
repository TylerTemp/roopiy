import logging
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


def extract_frames(target_path: str, temp_directory_path: str, temp_frame_quality: int = 0, fps: float = 30, ss: str | None = None, to: str | None = None, save_cut: str | None = None) -> bool:
    logger = logging.getLogger('roopiy.extract_frames')
    temp_frame_quality = temp_frame_quality * 31 // 100

    has_cut: bool = False
    image_command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-y'
    ]
    cut_command = list(image_command)
    if ss is not None:
        has_cut = True
        # image_command.extend(('-ss', ss))
        cut_command.extend(('-ss', ss))

    image_command.extend([
        '-hwaccel', 'auto',
        '-i', save_cut if has_cut else target_path,
        '-q:v', str(temp_frame_quality),
        '-pix_fmt', 'rgb24',
        '-vf', 'fps=' + str(fps),
    ])
    cut_command.extend([
        '-hwaccel', 'auto',
        '-i', target_path,
        '-strict', '-2'
    ])

    if to is not None:
        has_cut = True
        # image_command.extend(('-to', to))
        cut_command.extend(('-to', to))

    out_info = os.path.join(temp_directory_path, '%06d.png')
    image_command.append(out_info)
    cut_command.append(save_cut)

    if has_cut:
        assert save_cut

    if save_cut:
        logger.info('save %s - %s to %s', ss, to, save_cut)
        subprocess.check_output(cut_command)

    logger.info('images %s', image_command)
    subprocess.check_output(image_command)


def by_args(args: dict[str, str]):
    video_file: str = args['<video_file>']
    output_dir: str = args['<frames_dir>']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fps: int = int(args['--fps'])
    actual_fps: float = fps
    if fps == -1:
        actual_fps = detect_fps(video_file)

    extract_frames(video_file, output_dir, fps=actual_fps, ss=args['--ss'], to=args['--to'], save_cut=args['--save-cut'])
