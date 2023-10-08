import os
import pickle
import tqdm

import cv2
from insightface.app import FaceAnalysis

from roopiy.faces import FaceToDraw, draw_faces, load_face_analyser


def split_raw_faces(face_analyser: FaceAnalysis, target_raw_frames_folder: str, target_raw_faces_folder: str):
    # face_analyser = load_face_analyser()

    # target_raw_frames_folder = 'target_raw_frames'
    _, _, frame_images = next(os.walk(target_raw_frames_folder))
    # print(frame_images)

    # target_raw_faces_folder = 'target_raw_faces'
    if not os.path.isdir(target_raw_faces_folder):
        os.makedirs(target_raw_faces_folder)

    # all raw faces
    for frame_image_file in tqdm.tqdm(frame_images):
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


def by_args(args: dict[str, str]) -> None:
    input_dir = args['<frames_dir>']
    output_dir = args['<identify_dir>']

    model_root = args['--model-path'] or os.environ['ROOPIY_MODEL_PATH']

    face_analyser = load_face_analyser(model_root)

    split_raw_faces(face_analyser, input_dir, output_dir)
