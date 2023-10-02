import json
import logging
import os
import pickle
import typing
from dataclasses import dataclass

import cv2

from roopiy.faces import FaceToDraw, draw_faces, find_similar_face
from roopiy.utils import Face, draw_text_center


@dataclass
class ImageFileIndex:
    image_file: str
    index: int


@dataclass
class FaceGroupBasicInfo:
    image_file_indexes: list[ImageFileIndex]
    alias: str


class FrameFaceInfo(typing.TypedDict):
    target_alias: str | None


class FrameFacesInfo(typing.TypedDict):
    auto: bool
    faces: list[FrameFaceInfo]


def find_target_face(face: Face, target_alias_to_faces: dict[str, list[Face]], distance: float) -> str | None:
    for target_alias, target_faces in target_alias_to_faces.items():
        for target_face in target_faces:
            is_similar_face: bool = find_similar_face(face, target_face, distance)
            if is_similar_face:
                return target_alias
    return None


def group_faces(target_raw_frames_folder: str, target_raw_faces_folder: str, target_tagged_faces_folder: str, target_faces_groups: list[FaceGroupBasicInfo], distance: float):
    logger = logging.getLogger('roopiy.group_faces')

    if not os.path.isdir(target_tagged_faces_folder):
        os.makedirs(target_tagged_faces_folder)

    _, _, raw_faces_files = next(os.walk(target_raw_faces_folder))
    raw_faces_no_pk = [each for each in raw_faces_files if not each.endswith('.pk')]

    image_file_to_faces: dict[str, list[Face]] = {}

    for raw_faces_image_file in raw_faces_no_pk:
        raw_faces_pk_file = raw_faces_image_file + '.pk'
        logger.debug('checking %s', raw_faces_pk_file)
        with open(os.path.join(target_raw_faces_folder, raw_faces_pk_file), 'rb') as f:
            faces: list[Face] = pickle.load(f)
        image_file_to_faces[raw_faces_image_file] = faces

    # target faces info
    target_alias_to_faces : dict[str, list[Face]] = {}
    for each_target_basic_info in target_faces_groups:
        each_target_alias = each_target_basic_info.alias
        target_alias_to_faces[each_target_alias] = [
            image_file_to_faces[each.image_file][each.index]
            for each in each_target_basic_info.image_file_indexes
        ]
        # each_target_face = image_file_to_faces[each_target_basic_info.raw_faces_image_file][each_target_basic_info.in]

    # check each frame
    for image_file, faces in image_file_to_faces.items():
        frame_file = os.path.join(target_raw_frames_folder, image_file)

        frame = cv2.imread(frame_file)

        target_image_path = os.path.join(target_tagged_faces_folder, image_file)
        target_image_config = target_image_path + '.json'

        frame_faces_info: FrameFacesInfo = FrameFacesInfo(auto=True, faces=[])
        if os.path.isfile(target_image_config):
            with open(target_image_config, 'r', encoding='utf-8') as f:
                frame_faces_info_from_json = json.load(f)
            if not frame_faces_info_from_json.setdefault('auto', False):
                frame_faces_info = FrameFacesInfo(**frame_faces_info_from_json)
                logger.debug('manu %s: %s', image_file, frame_faces_info)
                for face_index, face in enumerate(faces):
                    target_alias = frame_faces_info['faces'][face_index]['target_alias']
                    if target_alias is None:
                        frame_faces_info['faces'].append(FrameFaceInfo(target_alias=None))
                        draw_faces(frame, [FaceToDraw(
                            face=face,
                            text=f'{face_index}',
                            color=(0, 0, 255),
                        )])
                    else:
                        frame_faces_info['faces'].append(FrameFaceInfo(target_alias=target_alias))
                        draw_faces(frame, [FaceToDraw(
                            face=face,
                            text=f'{face_index}|{target_alias}',
                            color=(255, 255, 255),
                        )])
                draw_text_center(frame, "LOCK", (0, 0, 255))
                cv2.imwrite(target_image_path, frame)
                continue

        for face_index, face in enumerate(faces):
            # check each target face
            target_alias = find_target_face(face, target_alias_to_faces, distance)
            if target_alias is None:
                frame_faces_info['faces'].append(FrameFaceInfo(target_alias=None))
                draw_faces(frame, [FaceToDraw(
                    face=face,
                    text=f'{face_index}',
                    color=(0, 0, 255),
                )])
            else:
                frame_faces_info['faces'].append(FrameFaceInfo(target_alias=target_alias))
                draw_faces(frame, [FaceToDraw(
                    face=face,
                    text=f'{face_index}|{target_alias}',
                    color=(255, 255, 255),
                )])

        logger.debug('auto %s: %s', image_file, frame_faces_info)

        cv2.imwrite(target_image_path, frame)
        with open(target_image_path+'.json', 'w', encoding='utf-8') as f:
            json.dump(frame_faces_info, f, indent=2)


def to_look(image_or_json: str) -> None:
    if not image_or_json.endswith('.json'):
        image_or_json += '.json'

    with open(image_or_json, 'r', encoding='utf-8') as f:
        config = json.load(f)
    if not config.get('auto', False):
        return

    config['auto'] = False
    image_path = os.path.splitext(image_or_json)[0]
    frame = cv2.imread(image_path)
    draw_text_center(frame, "LOCK", (0, 0, 255))
    # print(image_path)
    cv2.imwrite(image_path, frame)

    with open(image_or_json, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def by_args(args: dict[str, any]) -> None:
    # roopiy tag <frames_dir> <identify_dir> <tag_dir> [<group_face>]...
    need_lock: bool = args['lock']

    if need_lock:
        for image_or_json in args['<image_or_json>']:
            to_look(image_or_json)
        return

    frames_dir = args['<frames_dir>']
    identify_dir = args['<identify_dir>']
    tag_dir = args['<tag_dir>']
    distance = float(args['--distance'] or '0.85')

    face_group_basic_infos: list[FaceGroupBasicInfo] = []

    for group_face_args in args['<group_face>']:
        alias, image_index_map_str = group_face_args.split('/')
        image_file_and_index: list[tuple[str, int]] = []

        for image_index_map in image_index_map_str.split(','):
            image_file, index_str = image_index_map.split(':')
            image_file_and_index.append((image_file, int(index_str)))

        face_group_basic_infos.append(FaceGroupBasicInfo(
            alias=alias,
            image_file_indexes=[ImageFileIndex(image_file=image_file, index=index) for image_file, index in image_file_and_index]
        ))

    group_faces(frames_dir, identify_dir, tag_dir, face_group_basic_infos, distance)
