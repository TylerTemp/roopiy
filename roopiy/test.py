import os.path
import sys
import cv2
import insightface
from insightface.app.common import Face
import numpy
import typing
import pickle
import json
from gfpgan.utils import GFPGANer
from insightface.data import get_image as ins_get_image

Frame = numpy.ndarray[typing.Any, typing.Any]


delattr(Face, '__getattr__')


def enhance_face(face_enhancer: GFPGANer, target_face: Face, temp_frame: Frame) -> Frame:
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
    return temp_frame


def find_similar_face(face_1: Face, face_2: Face):
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
    return numpy.sum(numpy.square(face_1.normed_embedding - face_2.normed_embedding))


def main():
    face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    # model_path = resolve_relative_path()
    # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
    face_enhancer = GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=1, device='cuda')

    face_analyser.prepare(ctx_id=0)

    source_frame = cv2.imread("s.png")
    source_faces = face_analyser.get(source_frame)
    print(len(source_faces))
    source_face = source_faces[0]

    print(dir(source_face))
    sys.exit()

    with open('source_face.pcl', 'wb') as f:
        pickle.dump(source_face, f)

    with open('source_face.pcl', 'rb') as f:
        source_face = pickle.load(f)

    target_face_check: Face = None

    for target_img_path in ("t1.png", "t2.png", "t3.png"):
        print(target_img_path)
        target_frame = cv2.imread(target_img_path)
        target_faces = face_analyser.get(target_frame)
        # print(len(target_faces))
        if target_face_check is None:
            target_face_check = target_faces[0]
        for index, target_face in enumerate(target_faces):
            print(index, find_similar_face(target_face_check, target_face))

        # continue

        face_swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', providers=['CUDAExecutionProvider'])

        result = face_swapper.get(target_frame, target_faces[0], source_face, paste_back=True)
        cv2.imwrite(f'out_ori_{target_img_path}', result)

        enhanced_result = enhance_face(face_enhancer, target_faces[0], result)
        cv2.imwrite(f'out_enh_{target_img_path}', enhanced_result)

        to_draw_result = target_frame.copy()
        draw_result = face_analyser.draw_on(to_draw_result, target_faces)
        cv2.imwrite(f'out_drw_{target_img_path}', draw_result)


if __name__ == '__main__':
    main()
