from nevus_coco_model import NevusCocoClassifier
from nevus_model import NevusClassifier
import argparse
from pathlib import Path

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Sample Program")
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input JPEG image file"
    )

    parser.add_argument(
        "--nevus-model", type=Path, required=True, help="Path to nevus model"
    )

    parser.add_argument(
        "--nevus-coco-model", type=Path, required=True, help="Path to nevus-coco model"
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    nevus_coco_model = NevusCocoClassifier.load_from_checkpoint(
        args.nevus_coco_model
    ).cpu()
    nevus_classifier = NevusClassifier.load_from_checkpoint(args.nevus_model).cpu()

    for model in [nevus_coco_model, nevus_classifier]:
        model.eval()

    with open(args.input, "rb") as f:
        image_bytes = f.read()
        nevus_coco_score = nevus_coco_model.infer_image(image_bytes)
        if nevus_coco_score >= 0.5:
            print("Probably it's not correct image. Exit.")
        else:
            print(f"Hm, it's good image. Let's continue. Score: {nevus_coco_score}")
            prediction = nevus_classifier.infer_image(image_bytes)
            classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
            print("Classes: %s", classes)
            print("Prediction result: %s", prediction)

            winner_class = classes[prediction.index(max(prediction))]
            print(f"Result: {winner_class}")


if __name__ == "__main__":
    main()
