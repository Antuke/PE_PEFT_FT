"""Face detector, used only for the demo to crop faces, as datasets have already been face-cropped"""

import os
import cv2
import numpy as np


DEFAULT_FACE_DETECTOR = "utils/res10_300x300_ssd_iter_140000_fp16.caffemodel"
DEFAULT_DEPLOY = "utils/deploy.prototxt"


def enclosing_square(rect):
    # Crea un quadrato che contiene il rettangolo passato in ingresso
    x, y, w, h = rect
    side = max(w, h)
    # Centra il quadrato sulla bbox originale
    cx = x + w // 2
    cy = y + h // 2
    x_new = cx - side // 2
    y_new = cy - side // 2
    return (x_new, y_new, side, side)


def cut(frame, roi):
    pA = (int(roi[0]), int(roi[1]))
    pB = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))  # pB will be an internal point
    W, H = frame.shape[1], frame.shape[0]
    A0 = pA[0] if pA[0] >= 0 else 0
    A1 = pA[1] if pA[1] >= 0 else 0
    data = frame[A1 : pB[1], A0 : pB[0]]
    if pB[0] < W and pB[1] < H and pA[0] >= 0 and pA[1] >= 0:
        return data
    w, h = int(roi[2]), int(roi[3])
    img = np.zeros((h, w, frame.shape[2]), dtype=np.uint8)
    offX = int(-roi[0]) if roi[0] < 0 else 0
    offY = int(-roi[1]) if roi[1] < 0 else 0
    np.copyto(img[offY : offY + data.shape[0], offX : offX + data.shape[1]], data)
    return img


class FaceDetector:
    """Face detector to spot faces inside a picture."""

    def __init__(
        self,
        face_detector=DEFAULT_FACE_DETECTOR,
        deploy=DEFAULT_DEPLOY,
        confidence_threshold=0.8,
    ):
        self.detector = cv2.dnn.readNetFromCaffe(deploy, face_detector)
        self.confidence_threshold = confidence_threshold

    def detect(self, image, pad_rect=True):
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), [104, 117, 123], False, False
        )
        frameHeight, frameWidth, channels = image.shape
        self.detector.setInput(blob)
        detections = self.detector.forward()

        faces_result = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                f = (x1, y1, x2 - x1, y2 - y1)  # bbox: (x, y, w, h)
                if f[2] > 1 and f[3] > 1:
                    rect = enclosing_square(f) if pad_rect else f
                    img_crop = cut(image, rect)
                    if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
                        faces_result.append(
                            (img_crop, confidence, rect)
                        )  # usa rect (quadrato) come bbox finale
        if len(faces_result) == 0:
            return None
        return faces_result


if __name__ == "__main__":
    input_folder = "src/demo_images"
    output_crop_folder = "./test/detector/crop"
    output_bbox_folder = "./test/detector/bbox"

    os.makedirs(output_crop_folder, exist_ok=True)
    os.makedirs(output_bbox_folder, exist_ok=True)

    face_detector = FaceDetector(confidence_threshold=0.8)

    image_files = sorted(
        [
            f
            for f in os.listdir(input_folder)
            if os.path.isfile(os.path.join(input_folder, f))
        ]
    )

    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = face_detector.detect(img, pad_rect=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        if faces is not None:
            # Salva i crop dei volti
            for idx, (crop, confidence, bbox) in enumerate(faces):
                crop_path = os.path.join(
                    output_crop_folder, f"{base_name}_face{idx}.jpg"
                )
                cv2.imwrite(crop_path, crop)

            # Salva l'immagine originale con bbox quadrata
            img_bbox = img.copy()
            for idx, (_, _, bbox) in enumerate(faces):
                x, y, w, h = bbox  # bbox è già quadrata
                cv2.rectangle(
                    img_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2
                )  # rosso BGR
            bbox_path = os.path.join(output_bbox_folder, f"{base_name}_bbox.jpg")
            cv2.imwrite(bbox_path, img_bbox)
