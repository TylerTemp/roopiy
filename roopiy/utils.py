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


# class DataClassJSONEncoder(json.JSONEncoder):
#     def default(self, o):
#         if dataclasses.is_dataclass(o):
#             return dataclasses.asdict(o)
#         return super().default(o)

def draw_text_center(frame: Frame, text: str, color: tuple[int, int, int]):
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 2
    thickness = 4
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = int((frame.shape[1] - text_size[0]) / 2)
    text_y = int((frame.shape[0] + text_size[1]) / 2)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
