import json
import logging
import os
import json
import shutil

import cv2
import tqdm
from gfpgan import GFPGANer

from roopiy.faces import enhance_face, load_face_swapper, load_face_enhancer, load_face_analyser
from roopiy.tag import FrameFacesInfo
from roopiy.utils import Face, Frame, dict_to_face


def swap(face_swapper, face_enhancer: GFPGANer, alias_to_target_face: dict[str, Face], target_folder: str,
         target_raw_frames_folder: str, target_raw_faces_folder: str, target_tagged_faces_folder: str, swap_all: bool) -> None:
    logger = logging.getLogger('roopiy.swap')

    if swap_all:
        assert len(alias_to_target_face) == 1

    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    # tagged_faces_configs = glob.glob(os.path.join(target_tagged_faces_folder, '*.json'))
    _, _, tagged_faces_configs_raw = next(os.walk(target_tagged_faces_folder))
    tagged_faces_configs = [each for each in tagged_faces_configs_raw if each.endswith('.json')]

    for each_json_file in tqdm.tqdm(tagged_faces_configs):
        with open(os.path.join(target_tagged_faces_folder, each_json_file), 'r', encoding='utf-8') as f:
            config: FrameFacesInfo = json.load(f)

        faces_config = config['faces']

        image_file_name = os.path.splitext(each_json_file)[0]

        source_image_path = os.path.join(target_raw_frames_folder, image_file_name)
        replace_image_path = os.path.join(target_folder, image_file_name)

        if os.path.exists(replace_image_path):
            continue

        # if all(each['target_alias'] is None for each in faces_config):
        #     logger.debug('copy %s -> %s', image_file_name, replace_image_path)
        #     shutil.copy(source_image_path, replace_image_path)
        # else:

        frame: Frame | None = None
        frame_faces: list[Face] | None = None
        for index, face_config in enumerate(faces_config):
            target_alias = face_config['target_alias']
            target_face = alias_to_target_face.get(target_alias, list(alias_to_target_face.values())[0] if swap_all else None)
            if target_face is not None:
                if frame is None:
                    frame = cv2.imread(source_image_path)
                if frame_faces is None:
                    # load pickle
                    with open(os.path.join(target_raw_faces_folder, image_file_name + '.json'), 'rb') as f:
                        frame_faces = [dict_to_face(each) for each in json.load(f)]
                # order: from -> to face
                # print(image_file_name)
                try:
                    frame = face_swapper.get(frame, frame_faces[index], target_face, paste_back=True)
                except BaseException:
                    print(image_file_name)
                    raise
                enhance_face(face_enhancer, frame_faces[index], frame)

        if frame is None:
            logger.debug('copy %s -> %s', image_file_name, replace_image_path)
            shutil.copy(source_image_path, replace_image_path)
        else:
            logger.debug('swap %s -> %s', image_file_name, replace_image_path)
            cv2.imwrite(replace_image_path, frame)


def by_args(args: dict[str, any]) -> None:
    # <frames_dir> <tag_dir> <swap_dir>
    frames_dir = args['<frames_dir>']
    tag_dir = args['<tag_dir>']
    identify_dir = args['<identify_dir>']
    swap_dir = args['<swap_dir>']

    swap_maps = args['<swap_map>']

    model_root = args['--model-path'] or os.environ['ROOPIY_MODEL_PATH']

    face_swapper = load_face_swapper(model_root)
    face_enhancer = load_face_enhancer(model_root)

    alias_to_target_face: dict[str, Face] = {}
    face_analyser = None
    for swap_map in swap_maps:

        if face_analyser is None:

            face_analyser = load_face_analyser(model_root)

        alias, target_image_file = swap_map.split('/', 1)

        source_frame = cv2.imread(target_image_file)
        source_faces = face_analyser.get(source_frame)
        # print(len(source_faces))
        source_face = source_faces[0]
        alias_to_target_face[alias] = source_face

    swap(face_swapper, face_enhancer, alias_to_target_face, swap_dir,
         frames_dir, identify_dir, tag_dir, args['--all'])
