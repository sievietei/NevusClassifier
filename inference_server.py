from nevus_coco_model import NevusCocoClassifier
from nevus_model import NevusClassifier
import argparse
from pathlib import Path
import argparse

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image
import io


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Check if the requested URL is correct
        if self.path != "/api/analyse":
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"error": "Not Found"}
            self.wfile.write(json.dumps(response).encode())
            return

        # Check the content type
        content_type = self.headers["Content-Type"]
        if content_type != "image/jpeg":
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"error": "File is not a JPEG image"}
            self.wfile.write(json.dumps(response).encode())
            return

        # Get the content length
        content_length = int(self.headers["Content-Length"])

        # Read the image bytes
        image_bytes = self.rfile.read(content_length)
        nevus_coco_score = nevus_coco_model.infer_image(image_bytes)
        classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

        results = {
            "nevus_coco_score": nevus_coco_score,
            "classes": classes,
        }

        if nevus_coco_score < 0.5:
            prediction = nevus_classifier.infer_image(image_bytes)
            winner_class = classes[prediction.index(max(prediction))]

            results["prediction"] = prediction
            results["winner"] = winner_class

        # Send response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(results).encode())


def get_parser():
    parser = argparse.ArgumentParser(description="Sample Program")

    parser.add_argument(
        "--nevus-model", type=Path, required=True, help="Path to nevus model"
    )

    parser.add_argument(
        "--nevus-coco-model", type=Path, required=True, help="Path to nevus-coco model"
    )

    parser.add_argument("--port", type=int, required=False, default=8000, help="Port")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    nevus_coco_model = NevusCocoClassifier.load_from_checkpoint(
        args.nevus_coco_model
    ).cpu()
    nevus_classifier = NevusClassifier.load_from_checkpoint(args.nevus_model).cpu()

    for model in [nevus_coco_model, nevus_classifier]:
        model.eval()

    server_address = ("", args.port)
    server_class = HTTPServer
    httpd = server_class(server_address, RequestHandler)
    print(f"Starting server on port {args.port}...")
    httpd.serve_forever()
