"""
Usage:
    roopiy extract-video [video-options] <video_file> <frames_dir>
    roopiy identify <frames_dir> <identify_dir>
    roopiy tag lock <image_or_json>...
    roopiy tag [tag-options] <frames_dir> <identify_dir> <tag_dir> [<group_face>]...
    roopiy swap [--all] <frames_dir> <identify_dir> <tag_dir> <swap_dir> <swap_map>...
    roopiy create-video <swap_dir> <video_file> <output_video_file>

Video Options:
    --fps=<number>      use fps. -1 for no limit [default: 30]

Tag Options:
    -d, --distance=<number>     face distance for checking [default: 0.85]

Group Face:
    format: `ALIAS/FILE:INDEX,FILE:INDEX,FILE:INDEX...`
    e.g. `thanos/000005.png:1`, means 000005.png seconds face (count from 0) is thanos
    used for reference that you want to replace

Swap Map:
    format: `ALIAS/FILE`, will replace ALIAS with face in FILE
"""

import logging

import docpie

from roopiy import create_video
from roopiy import extract_video
from roopiy import identify
from roopiy import swap
from roopiy import tag


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    logger = logging.getLogger('roopiy')
    logger.setLevel(logging.DEBUG)

    docpie.logger.setLevel(logging.WARNING)
    pie = docpie.Docpie(__doc__, namedoptions=True)
    # pie.preview()
    # sys.exit()
    pie.docpie()

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


if __name__ == '__main__':
    main()
