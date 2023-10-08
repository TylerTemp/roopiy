import dataclasses
import json

import cv2
import insightface
from gfpgan import GFPGANer
from insightface.app.common import Face
import numpy
import typing

Frame = numpy.ndarray[typing.Any, typing.Any]

delattr(Face, '__getattr__')


class FaceJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Face):
            return o.__dict__
        if isinstance(o, numpy.ndarray):
            return {'value': o.tolist(), 'type': o.dtype.name}
        if isinstance(o, numpy.float32):
            return float(o)
        if isinstance(o, numpy.int64):
            return int(o)
        return super().default(o)


def dict_to_face(face_dict):
    new_dict = {}
    for key, value in face_dict.items():
        # if key not in ('det_score', 'gender', 'age'):
        #     value = numpy.array(value)
        # if key == 'det_score':
        #     value = numpy.float32(value)
        if key == 'age':
            pass
        elif key == 'gender':
            value = numpy.int64(value)
        elif key == 'det_score':
            value = numpy.float32(value)
        else:
            value = numpy.array(value['value'], numpy.dtype(value['type']))

        # print(key, value)
        new_dict[key] = value
    return Face(d=None, **new_dict)


def draw_text_center(frame: Frame, text: str, color: tuple[int, int, int]):
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 2
    thickness = 4
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = int((frame.shape[1] - text_size[0]) / 2)
    text_y = int((frame.shape[0] + text_size[1]) / 2)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
