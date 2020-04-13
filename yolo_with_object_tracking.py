from __future__ import print_function
import cv2
import argparse
import sys
import numpy as np
import os
from argparse import ArgumentParser
import json
from pprint import pprint
from random import randint
import itertools

(MAJOR_VER, MINOR_VER, SUBMINOR_VER) = (cv2.__version__).split(".")


class YoloPrediction:
    def __init__(self, *args, **kwargs):
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_width = 416
        self.input_height = 416
        self.image_name = kwargs.get("image")
        self.video_name = kwargs.get("video")
        self.folder_name = kwargs.get("folder")
        self.all_json_values = dict()
        self.tracker_types = [
            "BOOSTING",
            "MIL",
            "KCF",
            "TLD",
            "MEDIANFLOW",
            "GOTURN",
            "MOSSE",
            "CSRT",
        ]
        self.track_id = 1

        classes_file = os.path.join("yolov3", "traffic.names")
        self.classes = None
        with open(classes_file, "rt") as file:
            self.classes = file.read().rstrip("\n").split("\n")

        model_configuration = os.path.join("yolov3", "traffic_yolo.cfg")
        model_weights = os.path.join("yolov3", "traffic_yolo.weights")

        self.network = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # self.preprocess()
        # self.process_frame()

    def preprocess(self):
        self.capture = None
        if self.image_name:
            if not os.path.isfile(self.image_name):
                print("Input image file ", self.image_name, " doesn't exist")
                sys.exit(1)
            self.capture = cv2.VideoCapture(self.image_name)
        elif self.video_name:
            if not os.path.isfile(self.video_name):
                print("Input video file ", self.video_name, " doesn't exist")
                sys.exit(1)
            self.capture = cv2.VideoCapture(self.video_name)
        elif self.folder_name:
            if not os.path.exists(self.folder_name):
                print(
                    "Folder {} doesn't exist in the mentioned location".format(
                        self.folder_name
                    )
                )
                sys.exit(1)
            self.image_list = os.listdir(self.folder_name)

    def get_output_names(self):
        self.layer_names = self.network.getLayerNames()
        return [
            self.layer_names[i[0] - 1] for i in self.network.getUnconnectedOutLayers()
        ]

    def run_tracker(self, image, init_bbox, frame_list):
        tracker_types = ["CSRT", "KCF"]
        tracker_type = tracker_types[0]

        # Creating an object of the tracker
        if int(MAJOR_VER) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == "KCF":
                tracker = cv2.TrackerKCF_create()
            if tracker_type == "CSRT":
                tracker = cv2.TrackerCSRT_create()

        state = True
        bbox = init_bbox

        frame = cv2.imread(image)
        state = tracker.init(frame, bbox)

        counter = 0

        for frame_name in frame_list:
            frame = cv2.imread(os.path.join(self.folder_name, frame_name))

            if counter > len(frame_list):
                state = False

            state, bbox = tracker.update(frame)
            bbox = tuple(map(int, (i for i in bbox)))
            counter += 1

            if state == False:
                break

        return bbox, frame

    def calculate_iou(self, od_box, track_box):
        xA = max(od_box[0], track_box[0])
        yA = max(od_box[1], track_box[1])
        xB = min(od_box[2] + od_box[0], track_box[2] + track_box[0])
        yB = min(od_box[3] + od_box[1], track_box[3] + track_box[1])
        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        od_box_area = (od_box[2] + 1) * (od_box[3] + 1)
        track_box_area = (track_box[2] + 1) * (track_box[3] + 1)

        iou = inter_area / float(od_box_area + track_box_area - inter_area)
        return iou

    def process_frame(self):
        self.frame_count = 0
        if self.capture:
            while cv2.waitKey(1) < 0:

                has_frame, frame = self.capture.read()

                if not has_frame:
                    cv2.waitKey(3000)
                    break

                blob = cv2.dnn.blobFromImage(
                    frame,
                    1 / 255,
                    (self.input_width, self.input_height),
                    [0, 0, 0],
                    1,
                    crop=False,
                )

                self.network.setInput(blob)
                output_values = self.network.forward(self.get_output_names())
                self.postprocess(frame, output_values, {})

        elif self.image_list:
            self.detection_missed = 7
            image_values = list(map(int, [i[:-4] for i in self.image_list]))
            image_values.sort()
            for image in image_values:
                capture = cv2.VideoCapture(
                    os.path.join(self.folder_name, str(image) + ".jpg")
                )

                has_frame, frame = capture.read()
                if not has_frame:
                    break

                blob = cv2.dnn.blobFromImage(
                    frame,
                    1 / 255,
                    (self.input_width, self.input_height),
                    [0, 0, 0],
                    1,
                    crop=False,
                )

                if not self.all_json_values.keys():
                    self.network.setInput(blob)
                    output_values = self.network.forward(self.get_output_names())
                    self.postprocess(frame, output_values, {})

                else:
                    prev_image = image - 1
                    prev_image_size = str(
                        os.stat(
                            os.path.join(self.folder_name, str(prev_image) + ".jpg")
                        ).st_size
                    )

                    prev_image_id = str(prev_image) + ".jpg" + prev_image_size
                    regions = self.all_json_values.get(prev_image_id).get("regions")

                    tracked_boxes = dict()
                    for region in regions:
                        x = region.get("shape_attributes").get("x")
                        y = region.get("shape_attributes").get("y")
                        width = region.get("shape_attributes").get("width")
                        height = region.get("shape_attributes").get("height")
                        track_id = region.get("region_attributes").get("track_id")
                        class_id = self.classes.index(
                            region.get("region_attributes").get("class")
                        )

                        bbox = tuple((x, y, width, height))

                        new_tracked_box, track_frame = self.run_tracker(
                            os.path.join(self.folder_name, str(prev_image) + ".jpg"),
                            bbox,
                            [str(prev_image) + ".jpg", str(image) + ".jpg"],
                        )

                        tracked_boxes[(track_id, class_id)] = new_tracked_box

                    self.network.setInput(blob)
                    output_values = self.network.forward(self.get_output_names())
                    self.postprocess(frame, output_values, tracked_boxes)

        with open(self.folder_name + ".json", "w") as file:
            json.dump(self.all_json_values, file, indent=4)

    def postprocess(self, frame, output_values, tracked_boxes):
        self.class_ids = []
        self.confidences = []
        self.od_boxes = []
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        for output_value in output_values:
            for detection in output_value:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence_score = scores[class_id]

                if confidence_score > self.confidence_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    self.class_ids.append(class_id)
                    self.confidences.append(float(confidence_score))
                    self.od_boxes.append([left, top, width, height])

        image_size = str(
            os.stat(
                os.path.join(self.folder_name, str(self.frame_count) + ".jpg")
            ).st_size
        )

        image_id = str(self.frame_count) + ".jpg" + image_size

        indices = cv2.dnn.NMSBoxes(
            self.od_boxes,
            self.confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )

        output_list = []

        print(
            "Number of bboxes: {}, Image number: {}".format(
                len(indices), self.frame_count
            )
        )

        for i in indices:
            i = i[0]
            box = self.od_boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            output_list.append(
                "{} : {} : {}".format(
                    self.classes[self.class_ids[i]], self.confidences[i] * 100, box
                )
            )

            if not tracked_boxes.keys():
                region = [
                    {
                        "region_attributes": {
                            "class": self.classes[self.class_ids[i]],
                            "track_id": str(self.track_id),
                        },
                        "shape_attributes": {
                            "height": box[3],
                            "name": "rect",
                            "width": box[2],
                            "x": box[0],
                            "y": box[1],
                        },
                    }
                ]

                self.track_id += 1
                if image_id not in self.all_json_values.keys():
                    self.all_json_values[image_id] = {
                        "filename": str(self.frame_count) + ".jpg",
                        "size": image_size,
                        "regions": region,
                        "file_attributes": dict(),
                    }

                else:
                    self.all_json_values[image_id]["regions"].extend(region)

            else:
                iou_list = list()
                for key, value in tracked_boxes.items():
                    iou = self.calculate_iou(box, value)
                    iou_list.append(iou)

                if max(iou_list) > 0.5:
                    ind = iou_list.index(max(iou_list))

                    region_attributes, shape_attributes = (
                        list(tracked_boxes.keys())[ind],
                        list(tracked_boxes.values())[ind],
                    )

                    region = [
                        {
                            "region_attributes": {
                                "class": self.classes[region_attributes[1]],
                                "track_id": region_attributes[0],
                            },
                            "shape_attributes": {
                                "height": shape_attributes[3],
                                "name": "rect",
                                "width": shape_attributes[2],
                                "x": shape_attributes[0],
                                "y": shape_attributes[1],
                            },
                        }
                    ]

                    if image_id not in self.all_json_values.keys():
                        self.all_json_values[image_id] = {
                            "filename": str(self.frame_count) + ".jpg",
                            "size": image_size,
                            "regions": region,
                            "file_attributes": dict(),
                        }

                    else:
                        self.all_json_values[image_id]["regions"].extend(region)

                else:
                    region = [
                        {
                            "region_attributes": {
                                "class": self.classes[self.class_ids[i]],
                                "track_id": str(self.track_id),
                            },
                            "shape_attributes": {
                                "height": box[3],
                                "name": "rect",
                                "width": box[2],
                                "x": box[0],
                                "y": box[1],
                            },
                        }
                    ]

                    self.track_id += 1
                    if image_id not in self.all_json_values.keys():
                        self.all_json_values[image_id] = {
                            "filename": str(self.frame_count) + ".jpg",
                            "size": image_size,
                            "regions": region,
                            "file_attributes": dict(),
                        }

                    else:
                        self.all_json_values[image_id]["regions"].extend(region)

        self.frame_count += 1


parser = argparse.ArgumentParser(description="Object Detection using YOLO in OPENCV")
parser.add_argument("--image", help="Path to image file.")
parser.add_argument("--video", help="Path to video file.")
parser.add_argument("--folder", help="Path to folder of images.")
args = parser.parse_args()

if args.image:
    yolo_prediction = YoloPrediction(image=args.image)
elif args.video:
    yolo_prediction = YoloPrediction(video=args.video)
elif args.folder:
    yolo_prediction = YoloPrediction(folder=args.folder)
else:
    raise Exception("MissingArgumentError: Please define an argument")

yolo_prediction.preprocess()
yolo_prediction.process_frame()
