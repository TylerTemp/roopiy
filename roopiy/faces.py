import os.path
import typing

from gfpgan import GFPGANer
import insightface
from insightface.app import FaceAnalysis
from dataclasses import dataclass
from roopiy.utils import Face, Frame
import numpy
import cv2


@dataclass
class FaceInfo:
    face: Face
    # alias: str
    index: int


@dataclass
class FaceToDraw:
    face: Face
    text: str | None
    color: tuple[int, int, int]


def load_face_analyser(root: str) -> FaceAnalysis:
    face_analyser = FaceAnalysis(
        root=root,
        name='buffalo_l',
        providers=['CUDAExecutionProvider']
    )
    face_analyser.prepare(ctx_id=0)
    return face_analyser


class ChDir(object):

    cur_working_dir: str
    new_working_dir: str

    def __init__(self, new_working_dir: str):
        self.new_working_dir = new_working_dir

    def __enter__(self):
        self.cur_working_dir = os.getcwd()
        os.chdir(self.new_working_dir)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.chdir(self.cur_working_dir)


def load_face_enhancer(root: str) -> GFPGANer:
    # root_dir = os.path.normpath(os.path.abspath(os.path.join(__file__, '..', '..')))

    with ChDir(root):
        result = GFPGANer(model_path=os.path.join(root, 'GFPGANv1.4.pth'), upscale=1, device='cuda')

    return result


def load_face_swapper(root):
    # root_dir = os.path.normpath(os.path.abspath(os.path.join(__file__, '..', '..')))
    return insightface.model_zoo.get_model(
        # os.path.join(root_dir, 'temp/inswapper_128.onnx'),
        os.path.normpath(os.path.abspath(os.path.join(root, 'inswapper_128.onnx'))),
        root=root,
        download=False,
        providers=['CUDAExecutionProvider'])


def find_similar_face(face_1: Face, face_2: Face, expect_distance: float) -> bool:
    # many_faces = get_many_faces(frame)
    # if many_faces:
    #     for face in many_faces:
    #         if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
    #             distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
    #             if distance < roop.globals.similar_face_distance:
    #                 return face
    # return None
    # print(face_1.normed_embedding)
    # print(face_2.normed_embedding)
    distance = numpy.sum(numpy.square(face_1.normed_embedding - face_2.normed_embedding))
    # print(distance)
    return distance < expect_distance


def draw_faces(frame: Frame, faces: list[FaceToDraw]):
    for face_info in faces:
        face = face_info.face

        box = face.bbox.astype(int)
        # color = (0, 0, 255)
        # color = face_info.color

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), face_info.color, 2)

        # if face.kps is not None:
        #     kps = face.kps.astype(int)
        #     #print(landmark.shape)
        #     for l in range(kps.shape[0]):
        #         kps_color = (0, 0, 255)
        #         if l == 0 or l == 3:
        #             kps_color = (0, 255, 0)
        #         cv2.circle(frame, (kps[l][0], kps[l][1]), 1, kps_color,
        #                    2)

        # if face.gender is not None and face.age is not None:
        if face_info.text:
            cv2.putText(frame, face_info.text, (box[0] - 1, box[1] - 4),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        face_info.color, 1)

    # return frame


def enhance_face(face_enhancer: GFPGANer, target_face: Face, temp_frame: Frame) -> None:
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]
    if temp_face.size:
        _, _, temp_face = face_enhancer.enhance(
            temp_face,
            paste_back=True
        )
        temp_frame[start_y:end_y, start_x:end_x] = temp_face
    # return temp_frame
