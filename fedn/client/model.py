import os

import config
from fedn.utils.helpers.helpers import get_helper

from yolo_utils import (
    build_yolo_model,
    numpy_list_to_state_dict,
    state_dict_to_numpy_list,
)

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)


def compile_model(yolo_size: str = None, nc: int = None):
    size = yolo_size or config.YOLO_SIZE
    num_classes = nc or config.YOLO_NC
    return build_yolo_model(size, num_classes)


def save_parameters(model, out_path):
    arrays = state_dict_to_numpy_list(model.state_dict())
    helper.save(arrays, out_path)


def load_parameters(model_path, yolo_size: str = None, nc: int = None):
    size = yolo_size or config.YOLO_SIZE
    num_classes = nc or config.YOLO_NC
    arrays = helper.load(model_path)
    model = build_yolo_model(size, num_classes)
    state = numpy_list_to_state_dict(model, arrays)
    model.load_state_dict(state, strict=False)
    return model


def init_seed(out_path="seed.npz", yolo_size: str = None, nc: int = None):
    model = compile_model(yolo_size, nc)
    save_parameters(model, out_path)


if __name__ == "__main__":
    init_seed("../seed.npz")