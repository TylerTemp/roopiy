# Roopiy #

A video face swap tool using insightface

## Acceleration ##

1.  Install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2.  Download [cuDNN for Cuda 11.x](https://developer.nvidia.com/rdp/cudnn-archive)
3.  For windows go `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{YourVersion}`, copy the `bin`, `include`, `lib` folders from cuDNN to the CUDA folder, and add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{YourVersion}\bin` to path

## Installation ##

1.  clone this project
2.  `pip install -e .`.
3.  change the `torch` and `torchvision` in `requirements.txt` as you need, `pip uninstall -y onnxruntime` then `pip install -r requirements.txt`
4.  Setup a model path, e.g. `temp` (note: use absolute path!)
5.  download `inswapper_128.onnx` into your model path, from [huggingface](https://huggingface.co/ezioruan/inswapper_128.onnx) or [here](https://static.notexists.top/mirror/insightface/inswapper_128.onnx)
6.  download [buffalo_l.zip](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) and unzip under `temp/models/buffalo_l`; you folder now looks like: `temp/models/buffalo_l/*.onnx`
7.  download [GFPGANv1.4.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth) into your model path
8.  download [detection_Resnet50_Final.pth](https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth) into your model path `gfpgan/weights`
9.  download [parsing_parsenet.pth](https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth) into your model path `gfpgan/weights`

## Model Path ##

either use `ROOPIY_MODEL_PATH={PathToYourModel} roopiy` or `roopiy --model-path={PathToYourModel}` to set the model path

## Basic Run ##

use `roopiy` to run. If it does not work, run `python -m roopiy`

check `roopiy --help` for construction
