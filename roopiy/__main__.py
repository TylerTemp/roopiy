"""
Usage:
    roopiy [global-options] extract-video [video-options] <video_file> <frames_dir>
    roopiy [global-options] identify <frames_dir> <identify_dir>
    roopiy [global-options] tag lock <image_or_json>...
    roopiy [global-options] tag [tag-options] <frames_dir> <identify_dir> <tag_dir> [<group_face>]...
    roopiy [global-options] swap [--all] <frames_dir> <identify_dir> <tag_dir> <swap_dir> <swap_map>...
    roopiy [global-options] create-video <swap_dir> <video_file> <output_video_file>
    roopiy [global-options] image [image-options] <target_face_image> <source_image> [<output_image>]

Global Options:
    --log=<level>       log level [default: INFO]
    --model-path=<model_path>    model path you downloaded

Video Options:
    --fps=<number>      use fps. -1 for no limit [default: 30]
    -s, --ss=<start_time>
    -t, --to=<crop_length>
    -c, --save-cut=<cut_video>

Tag Options:
    -d, --distance=<number>     face distance for checking [default: 0.85]

Image Options:
    -s, --source-index=<number>     source image face index [default: 0]
    -t, --target-index=<number>     target image face index [default: 0]
    --no-enhance, --ne              disable face enhancement

Group Face:
    format: `ALIAS/FILE:INDEX,FILE:INDEX,FILE:INDEX...`
    e.g. `thanos/000005.png:1`, means 000005.png seconds face (count from 0) is thanos
    used for reference that you want to replace

Swap Map:
    format: `ALIAS/FILE`, will replace ALIAS with face in FILE
"""

import logging
import warnings
import sys
with warnings.catch_warnings():
    import torchvision

import docpie

from roopiy import create_video
from roopiy import extract_video
from roopiy import identify
from roopiy import swap
from roopiy import tag


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARNING)

    logger = logging.getLogger('roopiy')

    docpie.logger.setLevel(logging.WARNING)
    pie = docpie.Docpie(__doc__, namedoptions=True)
    # pie.preview()
    # sys.exit()
    pie.docpie()

    logger.setLevel(pie['--log'])

    if pie['extract-video']:
        extract_video.by_args(pie)
        return

    if pie['identify']:
        identify.by_args(pie)
        return

    if pie['tag']:
        # print(pie)
        # sys.exit()
        tag.by_args(pie)
        return

    if pie['swap']:
        swap.by_args(pie)
        return

    if pie['create-video']:
        create_video.by_args(pie)
        return

    if pie['image']:
        swap.image_by_args(pie)
        return

    raise NotImplementedError(' '.join(sys.argv[1:]))


if __name__ == '__main__':
    main()
