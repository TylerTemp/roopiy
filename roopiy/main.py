"""
Usage:
    roopiy [<group_face>]...
"""

import json
import logging
import os
import pickle
import shutil
import typing
from dataclasses import dataclass
# import glob

import cv2
import docpie
from gfpgan import GFPGANer
from insightface.app import FaceAnalysis

from roopiy.faces import draw_faces, load_face_analyser, find_similar_face, FaceToDraw, enhance_face, \
    load_face_enhancer, load_face_swapper
from roopiy.utils import Face, Frame


def split_raw_faces(face_analyser: FaceAnalysis, target_raw_frames_folder: str, target_raw_faces_folder: str):
    # face_analyser = load_face_analyser()

    # target_raw_frames_folder = 'target_raw_frames'
    _, _, frame_images = next(os.walk(target_raw_frames_folder))
    print(frame_images)

    # target_raw_faces_folder = 'target_raw_faces'
    if not os.path.isdir(target_raw_faces_folder):
        os.makedirs(target_raw_faces_folder)

    # all raw faces
    for frame_image_file in frame_images:
        frame = cv2.imread(os.path.join(target_raw_frames_folder, frame_image_file))
        faces = face_analyser.get(frame)
        faces_to_draw: list[FaceToDraw] = [FaceToDraw(
            face=face,
            text=str(index),
            color=(0, 255, 0)) for index, face in enumerate(faces)]
        drew_frame = frame.copy()
        draw_faces(drew_frame, faces_to_draw)
        target_raw_faces_path = os.path.join(target_raw_faces_folder, frame_image_file)
        cv2.imwrite(target_raw_faces_path, drew_frame)

        with open(target_raw_faces_path + '.pk', 'wb') as f:
            pickle.dump(faces, f)


@dataclass
class ImageFileIndex:
    image_file: str
    index: int


@dataclass
class FaceGroupBasicInfo:
    image_file_indexes: list[ImageFileIndex]
    alias: str


# @dataclass
# class FaceGroupInfo:
#     faces: list[Face]
#     alias: str


@dataclass
class FaceLoaded:
    face: Face
    image_file: str
    index: int
    alias: str | None


@dataclass
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


def swap(face_swapper, face_enhancer: GFPGANer, alias_to_target_face: dict[str, Face], target_folder: str,
         target_raw_frames_folder: str, target_raw_faces_folder: str, target_tagged_faces_folder: str):
    logger = logging.getLogger('roopiy.swap')
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    # tagged_faces_configs = glob.glob(os.path.join(target_tagged_faces_folder, '*.json'))
    _, _, tagged_faces_configs_raw = next(os.walk(target_tagged_faces_folder))
    tagged_faces_configs = [each for each in tagged_faces_configs_raw if each.endswith('.json')]

    for each_json_file in tagged_faces_configs:
        with open(os.path.join(target_tagged_faces_folder, each_json_file), 'r', encoding='utf-8') as f:
            config: FrameFacesInfo = json.load(f)

        faces_config = config['faces']

        image_file_name = os.path.splitext(each_json_file)[0]

        source_image_path = os.path.join(target_raw_frames_folder, image_file_name)
        replace_image_path = os.path.join(target_folder, image_file_name)

        if all(each['target_alias'] is None for each in faces_config):
            logger.debug('copy %s -> %s', image_file_name, replace_image_path)
            shutil.copy(source_image_path, replace_image_path)
        else:
            frame: Frame | None = None
            frame_faces: list[Face] | None = None
            for index, face_config in enumerate(faces_config):
                target_alias = face_config['target_alias']
                target_face = alias_to_target_face.get(target_alias, None)
                if target_face is not None:
                    if frame is None:
                        frame = cv2.imread(source_image_path)
                    if frame_faces is None:
                        # load pickle
                        with open(os.path.join(target_raw_faces_folder, image_file_name + '.pk'), 'rb') as f:
                            frame_faces = pickle.load(f)
                    # order: from -> to face
                    frame = face_swapper.get(frame, frame_faces[index], target_face, paste_back=True)
                    enhance_face(face_enhancer, frame_faces[index], frame)

            if frame is None:
                logger.debug('copy %s -> %s', image_file_name, replace_image_path)
                shutil.copy(source_image_path, replace_image_path)
            else:
                logger.debug('swap %s -> %s', image_file_name, replace_image_path)
                cv2.imwrite(replace_image_path, frame)


def main():
    face_analyser = load_face_analyser()
    face_swapper = load_face_swapper()
    face_enhancer = load_face_enhancer()
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    logger = logging.getLogger('roopiy')
    logger.setLevel(logging.DEBUG)

    docpie.logger.setLevel(logging.WARNING)
    pie = docpie.docpie(__doc__)

    target_raw_frames_folder = 'target_raw_frames'
    target_raw_faces_folder = 'target_raw_faces'
    split_raw_faces(face_analyser, target_raw_frames_folder, target_raw_faces_folder)

    face_group_basic_infos: list[FaceGroupBasicInfo] = []

    for group_face_args in pie['<group_face>']:
        alias, image_index_map_str = group_face_args.split('/')
        image_file_and_index: list[tuple[str, int]] = []

        for image_index_map in image_index_map_str.split(','):
            image_file, index_str = image_index_map.split(':')
            image_file_and_index.append((image_file, int(index_str)))

        face_group_basic_infos.append(FaceGroupBasicInfo(
            alias=alias,
            image_file_indexes=[ImageFileIndex(image_file=image_file, index=index) for image_file, index in image_file_and_index]
        ))

    target_tagged_faces_folder = 'target_tagged_faces'
    group_faces(target_raw_frames_folder, target_raw_faces_folder, target_tagged_faces_folder, face_group_basic_infos, 0.85)

    source_frame = cv2.imread("s.png")
    source_faces = face_analyser.get(source_frame)
    print(len(source_faces))
    source_face = source_faces[0]

    target_folder = "target_replaced"

    swap(face_swapper, face_enhancer, {'luck': source_face},
         target_folder, target_raw_frames_folder, target_raw_faces_folder, target_tagged_faces_folder)


if __name__ == '__main__':
    main()
