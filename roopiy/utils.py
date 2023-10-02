import dataclasses
import json

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
