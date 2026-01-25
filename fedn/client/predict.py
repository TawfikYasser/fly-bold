import argparse
import os
import sys
from pathlib import Path

import config
from fedn.utils.helpers.helpers import save_metrics

from data import resolve_client_yaml
from model import load_parameters
from validate import run_yolo_val
from yolo_utils import count_images_from_yaml, save_state_dict_as_yolo_checkpoint


def predict(in_model_path, out_json_path, client_id: int, data_root: str, yolo_size: str, img: int, nc: int):
    data_yaml = resolve_client_yaml(data_root, client_id)

    model = load_parameters(in_model_path, yolo_size=yolo_size, nc=nc)
    tmp_weights = Path(out_json_path).with_suffix(".pt")
    save_state_dict_as_yolo_checkpoint(model.state_dict(), yolo_size, tmp_weights, nc)

    metrics = run_yolo_val(str(tmp_weights), str(data_yaml), img)
    metrics.setdefault("client_id", client_id)
    train_images, val_images = count_images_from_yaml(str(data_yaml))
    metrics.setdefault("num_val_examples", val_images)
    metrics.setdefault("num_train_examples", train_images)
    metrics.setdefault("note", "Prediction reuses YOLO validation metrics for simplicity.")
    save_metrics(metrics, out_json_path)


def parse_args():
    parser = argparse.ArgumentParser(description="FEDn YOLOv5 predict entrypoint")
    parser.add_argument("in_model", help="Input model weights")
    parser.add_argument("out_json", help="Output JSON path")
    parser.add_argument("--client-id", type=int, default=config.CLIENT_INDEX)
    parser.add_argument("--data-root", default=config.DATA_ROOT)
    parser.add_argument("--yolo-size", default=config.YOLO_SIZE)
    parser.add_argument("--img", type=int, default=config.IMG_SIZE)
    parser.add_argument("--nc", type=int, default=config.YOLO_NC)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(
        args.in_model,
        args.out_json,
        args.client_id,
        args.data_root,
        args.yolo_size,
        args.img,
        args.nc,
    )