import os
import subprocess


def create_video(target_path: str, temp_directory_path: str, output_video_quality: int = 35, fps: float = 30) -> None:
    output_video_quality = (output_video_quality + 1) * 51 // 100
    commands = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-hwaccel', 'auto',
        '-r', str(fps),
        '-i', os.path.join(temp_directory_path, '%06d.png'),
        '-c:v', 'libx264',
        '-crf', str(output_video_quality),
        '-pix_fmt', 'yuv420p',
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1',
        '-y', target_path
    ]
    # if roop.globals.output_video_encoder in ['libx264', 'libx265', 'libvpx']:
    #     commands.extend(['-crf', str(output_video_quality)])
    # if roop.globals.output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
    #     commands.extend(['-cq', str(output_video_quality)])
    # commands.extend(['-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])
    subprocess.check_output(commands)


def restore_audio(silent_video: str, sound_video: str, output_path: str) -> None:
    commands = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-hwaccel', 'auto',
        '-i', silent_video,
        '-i', sound_video,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-y',
        output_path
    ]

    subprocess.check_output(commands)
    # temp_output_path = get_temp_output_path(target_path)
    # done = run_ffmpeg(['-i', temp_output_path, '-i', target_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path])
    # if not done:
    #     move_temp(target_path, output_path)


def by_args(args: dict[str, str]) -> None:
    # <swap_dir> <video_file> <output_video_file>
    swap_dir = args['<swap_dir>']
    ori_video_file = args['<video_file>']
    output_video_file = args['<output_video_file>']
    output_folder, output_file = os.path.split(output_video_file)
    silent_video = os.path.join(output_folder, f's_{output_file}')
    create_video(silent_video, swap_dir)
    restore_audio(silent_video, ori_video_file, output_video_file)
    os.remove(silent_video)
