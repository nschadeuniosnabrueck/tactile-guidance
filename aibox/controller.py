"""
This script is using code from the following sources:
- YOLOv5 🚀 by Ultralytics, AGPL-3.0 license, https://github.com/ultralytics/yolov5
- StrongSORT MOT, https://github.com/dyhBUPT/StrongSORT, https://pypi.org/project/strongsort/
- Youtube Tutorial "Simple YOLOv8 Object Detection & Tracking with StrongSORT & ByteTrack" by Nicolai Nielsen, https://www.youtube.com/watch?v=oDALtKbprHg
- https://github.com/zenjieli/Yolov5StrongSORT/blob/master/track.py, original: https://github.com/mikel-brostrom/yolo_tracking/commit/9fec03ddba453959f03ab59bffc36669ae2e932a
"""

import sys
import json
from pathlib import Path

# Use the project file packages instead of the conda packages, i.e. add to system path for import
file = Path(__file__).resolve()
root = file.parents[0]
sys.path.append(str(root) + '/yolov5')
sys.path.append(str(root) + '/strongsort')
sys.path.append(str(root) + '/unidepth')
sys.path.append(str(root) + '/midas')

# Utility
import time
from datetime import datetime
import pandas as pd
import numpy as np
import threading
import queue
from playsound import playsound

# Image processing
import cv2

# Object tracking
import torch
from labels import coco_labels  # COCO labels dictionary
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                                  increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, xywh2xyxy)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from strongsort.strong_sort import StrongSORT  # there is also a pip install, but it has multiple errors
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend

# Depth Estimation
import unidepth_estimator
from unidepth_estimator import UniDepthEstimator  # metric
from midas_estimator import MidasDepthEstimator  # relative
from midas.run import create_side_by_side


def _deep_merge_missing(target: dict, source: dict):
    """Recursively copy keys from `source` into `target` without overwriting existing keys."""
    for k, v in source.items():
        if k not in target:
            target[k] = v
        elif isinstance(target[k], dict) and isinstance(v, dict):
            _deep_merge_missing(target[k], v)


def _patch_unidepth_v2(model_type: str):
    """
    Patches the exact `UniDepthV2` class used by `unidepth_estimator.py`
    (`unidepth_estimator.UniDepthV2`) so that HuggingFace Hub's updated `config.json`
    (which omits `config["data"]["shape_constraints"]`) includes all keys required by
    `unidepthv2.py` (`pixels_bounds`, `pixels_bounds_ori`, `ratio_bounds`, `patch_size`).
    """
    # Collect all loaded UniDepthV2 class references across module namespaces
    target_classes = set()
    if hasattr(unidepth_estimator, "UniDepthV2"):
        target_classes.add(unidepth_estimator.UniDepthV2)
    try:
        from unidepth.models.unidepthv2.unidepthv2 import UniDepthV2 as UDV2_A
        target_classes.add(UDV2_A)
    except Exception:
        pass
    try:
        from unidepth.unidepth.models.unidepthv2.unidepthv2 import UniDepthV2 as UDV2_B
        target_classes.add(UDV2_B)
    except Exception:
        pass

    # Locate local config JSON in aibox/unidepth/configs/ if available
    local_cfg = {}
    configs_dir = root / "unidepth" / "configs"
    if configs_dir.is_dir():
        normalized_name = model_type.replace("-", "_")  # e.g. 'v2_vits14'
        candidates = list(configs_dir.glob(f"*{normalized_name}*.json")) + list(configs_dir.glob("*v2*.json"))
        for cfg_path in candidates:
            try:
                with open(cfg_path, "r") as f:
                    local_cfg = json.load(f)
                break
            except Exception:
                continue

    for cls in target_classes:
        if getattr(cls, "_aibox_patched", False):
            continue

        orig_build = cls.build

        def _make_safe_build(_orig):
            def _safe_build(self, config):
                if isinstance(config, dict):
                    if local_cfg:
                        _deep_merge_missing(config, local_cfg)
                    if "data" not in config or not isinstance(config["data"], dict):
                        config["data"] = {}

                    encoder_name = (
                        config.get("model", {})
                        .get("pixel_encoder", {})
                        .get("name", "dinov2")
                    )
                    patch_size = 14 if "dino" in encoder_name else 16

                    default_sc = {
                        "pixels_bounds": [1400, 2400],
                        "pixels_bounds_ori": [1400, 2400],
                        "ratio_bounds": [0.5, 2.5],
                        "patch_size": patch_size,
                    }
                    existing_sc = config["data"].get("shape_constraints")
                    if isinstance(existing_sc, dict):
                        merged_sc = {**default_sc, **existing_sc}
                    else:
                        merged_sc = dict(default_sc)
                    config["data"]["shape_constraints"] = merged_sc

                return _orig(self, config)

            return _safe_build

        cls.build = _make_safe_build(orig_build)

        # Guard `from_pretrained` against strict state_dict mismatches from newer HF checkpoints
        orig_from_pretrained = cls.from_pretrained

        def _make_safe_from_pretrained(_orig_fp):
            def _safe_from_pretrained(*args, **kwargs):
                try:
                    return _orig_fp(*args, **kwargs)
                except RuntimeError as err:
                    if "state_dict" in str(err) and "strict" not in kwargs:
                        print("Retrying UniDepthV2.from_pretrained with strict=False...")
                        kwargs["strict"] = False
                        return _orig_fp(*args, **kwargs)
                    raise

            return _safe_from_pretrained

        cls.from_pretrained = _make_safe_from_pretrained(orig_from_pretrained)
        cls._aibox_patched = True


def beginning_sound():
    file = 'resources/sound/beginning.mp3'
    playsound(str(file))


def play_start():
    play_start_thread = threading.Thread(target=beginning_sound, name='play_start')
    play_start_thread.start()


def key_listener(key_queue=None):
    """Thread function to listen for input for each trial."""
    while True:
        key = cv2.waitKey(50)
        """
        key = input()  # Blocks until the user inputs a key
        if key in ['s', 'y', 'n', 'c']:  # Accept only 'y' or 'n'
            key_queue.put((key, datetime.now()))  # Store key and timestamp
        """
        return key  # Exit after receiving valid input


def bbs_to_depth(image, depth=None, bbs=None):
    """
    Calculate the mean depth for bounding boxes (BBs) in an image.

    Args:
        image (numpy.ndarray): The input image.
        depth (numpy.ndarray, optional): The depth map corresponding to the input image. Defaults to None.
        bbs (list of lists, optional): A list of bounding boxes, where each bounding box is represented 
                                    as a list/array of 8 values [x_center, y_center, w, h, track_id, cls, conf, mean_depth].

    Returns:
        list: A list of bounding boxes with updated mean depth values.
    """
    if bbs is None or len(bbs) == 0:
        return []

    if depth is None:
        return list(bbs)

    outputs = []
    h_img, w_img = depth.shape[:2]

    for bb in bbs:
        bb = np.array(bb, dtype=float)
        if bb[7] == -1:  # Calculate depth if not already populated
            xc, yc, bw, bh = [int(round(coord)) for coord in bb[:4]]
            x1 = max(0, xc - bw // 2)
            x2 = min(w_img, xc + max(1, bw // 2))
            y1 = max(0, yc - bh // 2)
            y2 = min(h_img, yc + max(1, bh // 2))

            roi = depth[y1:y2, x1:x2]
            if roi.size > 0:
                valid = roi[np.isfinite(roi)]
                mean_depth = float(np.mean(valid)) if valid.size > 0 else -1.0
                bb[7] = mean_depth
            else:
                bb[7] = -1.0
        outputs.append(bb)

    return outputs


def close_app(controller):
    """
    Closes the application by stopping the controller's vibration, destroying all OpenCV windows,
    stopping all running threads, disconnecting the controller's belt, and exiting the program.

    Args:
        controller: An instance of the controller that manages the vibration and belt connection.
                    If None, the function will skip the vibration stop and belt disconnection steps.
    """
    controller.stop_vibration() if controller else None
    cv2.destroyAllWindows()

    # Release locks and stop background threads
    threads = threading.enumerate()
    for thread in threads:
        if thread is not threading.current_thread():
            with suppress_exceptions():
                thread._tstate_lock = None
                thread._stop()

    controller.disconnect_belt() if controller else None
    print("Application will be closed.")
    sys.exit()


class suppress_exceptions:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class AutoAssign:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TaskController(AutoAssign):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.variables = ['object_class', 'start_time', 'navigation_time', 'freezing_time', 'grasping_time', 'end_time', 'key']

    def save_output_data(self):
        # Fill missing values with NA for reshaping
        n = len(self.variables)
        if len(self.output_data) % n != 0:
            missing_data = (n - len(self.output_data) % n) * ['NA']
            self.output_data.extend(missing_data)

        if len(self.output_data) > 0:
            df = pd.DataFrame(np.array(self.output_data).reshape(len(self.output_data) // n, n), columns=self.variables)
            print(df)
            df.to_csv(self.output_path + f"{self.condition}_participant_{self.participant}.csv")

    def load_object_detector(self):

        print(f'\nLOADING OBJECT DETECTORS')

        self.device = select_device(self.device)
        self.model_obj = DetectMultiBackend(self.weights_obj, device=self.device, dnn=self.dnn, fp16=self.half)
        # self.model_obj = AutoBackend(self.weights_obj, device=self.device, dnn=self.dnn, fp16=self.half)
        # self.model_obj = YOLO(self.weights_obj, task='detect')
        # self.model_obj.to('cuda')
        self.model_hand = DetectMultiBackend(self.weights_hand, device=self.device, dnn=self.dnn, fp16=self.half)

        self.names_obj = self.model_obj.names
        # self.stride_obj, self.names_obj, self.pt_obj = self.model_obj.stride, self.model_obj.names, self.model_obj.pt
        self.stride_hand, self.names_hand, self.pt_hand = self.model_hand.stride, self.model_hand.names, self.model_hand.pt
        # self.imgsz = check_img_size(self.imgsz, s=self.stride_obj) # check image size
        self.dt = (Profile(), Profile(), Profile())

        print(f'\nOBJECT DETECTORS LOADED SUCCESFULLY')

    def load_object_tracker(self, max_age=70, n_init=3):

        print(f'\nLOADING OBJECT TRACKER')

        self.tracker = StrongSORT(
            model_weights=self.weights_tracker,
            device=self.device,
            fp16=False,
            max_dist=0.5,          # The matching threshold. Samples with larger distance are considered an invalid match
            max_iou_distance=0.7,  # Gating threshold. Associations with cost larger than this value are disregarded.
            max_age=max_age,       # Maximum number of missed misses (prediction calls, i.e. frames) before a track is deleted
            n_init=n_init,         # Number of frames that a track remains in initialization phase --> if 0, track is confirmed on first detection
            nn_budget=100,         # Maximum size of the appearance descriptors gallery
            mc_lambda=0.995,       # matching with both appearance (1 - MC_LAMBDA) and motion cost
            ema_alpha=0.9          # updates appearance state in an exponential moving average manner
        )

        print(f'\nOBJECT TRACKER LOADED SUCCESFULLY')

    def load_depth_estimator(self):

        print(f'\nLOADING DEPTH ESTIMATOR')

        if self.metric:
            _patch_unidepth_v2(self.weights_depth_estimator)
            self.depth_estimator = UniDepthEstimator(
                model_type=self.weights_depth_estimator,  # v2-vits14, v1-cnvnxtl
                device=self.device
            )
        else:
            self.depth_estimator = MidasDepthEstimator(
                model_type=self.weights_depth_estimator,  # midas_v21_384, dpt_levit_224
                device=self.device
            )

        print(f'\nDEPTH ESTIMATOR LOADED SUCCESFULLY')

    def warmup_model(self, model, type='detector'):

        print(f'\nWARMING UP MODEL...')

        if type == 'detector':
            # model.warmup(imgsz=(1 if self.pt_obj or self.model_obj.triton else self.bs, 3, *self.imgsz))
            model.warmup(imgsz=(1 if self.pt_hand or self.model_hand.triton else self.bs, 3, *self.imgsz))

        if type == 'tracker':
            model.warmup()

    def _predict_and_normalize_depth(self, im0):
        """Run depth estimator and guarantee a 2D float32 numpy array matching `im0.shape[:2]`."""
        depthmap, _ = self.depth_estimator.predict_depth(im0)
        if isinstance(depthmap, torch.Tensor):
            depthmap = depthmap.detach().cpu().numpy()
        depthmap = np.squeeze(np.asarray(depthmap, dtype=np.float32))
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        if depthmap.ndim == 2 and depthmap.shape[:2] != im0.shape[:2]:
            depthmap = cv2.resize(depthmap, (im0.shape[1], im0.shape[0]), interpolation=cv2.INTER_NEAREST)
        return depthmap

    def get_depth(self, im0, frame, outputs, prev_outputs, depthmap=None, frame_factor=10):
        """
        Estimate and update depth information for given frames.
        """
        if frame % frame_factor == 0 or depthmap is None:
            depthmap = self._predict_and_normalize_depth(im0)
            outputs = bbs_to_depth(im0, depthmap, outputs)
        else:
            if prev_outputs.size > 0 and len(outputs) > 0:
                for output in outputs:
                    if output[4] != -1:  # tracking ID
                        match = prev_outputs[prev_outputs[:, 4] == output[4]]
                    else:  # class number
                        match = prev_outputs[prev_outputs[:, 5] == output[5]]
                    if match.size > 0:
                        output[7] = match[0][7]
                    else:
                        output[7] = -1
            # Calculate depth from cached depthmap for any newly appeared bounding boxes
            outputs = bbs_to_depth(im0, depthmap, outputs)

        return depthmap, outputs

    def experiment_trial_logic(self, pressed_key):
        """
        Handles the logic for each trial in the experiment based on the pressed key.
        """
        # end trial
        if pressed_key in [ord('y'), ord('n'), ord('f'), ord('t')] and not self.ready_for_next_trial:

            trial_end_time = time.time()
            self.output_data.append(trial_end_time)
            self.output_data.append(self.bracelet_controller.navigation_time)
            self.output_data.append(self.bracelet_controller.freezing_time)
            self.output_data.append(self.bracelet_controller.grasping_time)
            self.bracelet_controller.navigation_time = 'NA'
            self.bracelet_controller.freezing_time = 'NA'
            self.bracelet_controller.grasping_time = 'NA'

            self.output_data.append(chr(pressed_key))

            self.classes_obj = self.orig_classes_obj

            self.bracelet_controller.frozen = False

            if pressed_key == ord('y'):
                print("TRIAL SUCCESSFUL")
            elif pressed_key == ord('n'):
                print("TRIAL FAILED")
            elif pressed_key == ord('f'):
                print("SYSTEM FAILED")
            elif pressed_key == ord('t'):
                print("WRONG TARGET")

            if self.obj_index >= len(self.target_objs) - 1:
                print("ALL TARGETS COVERED")
                self.save_output_data()
                return "break"
            else:
                print("MOVING TO NEXT TARGET")
                self.obj_index += 1
                self.ready_for_next_trial = True
                self.class_target_obj = -1

        # start next trial
        elif pressed_key == ord('s') and self.ready_for_next_trial:
            print("STARTING NEXT TRIAL")
            trial_start_time = time.time()
            self.output_data.append(trial_start_time)
            self.target_entered = False
            self.ready_for_next_trial = False
            self.bracelet_controller.vibrate = True

        # end experiment
        elif pressed_key == ord('c'):  # 'q' interferes with opencv

            self.output_data.append(self.bracelet_controller.navigation_time)
            self.output_data.append(self.bracelet_controller.freezing_time)
            self.output_data.append(self.bracelet_controller.grasping_time)

            self.output_data.append(chr(pressed_key))

            self.save_output_data()

            if self.belt_controller:
                self.belt_controller.stop_vibration()
            return "break"

    def experiment_loop(self, save_dir, save_img, index_add, vid_path, vid_writer):
        """
        Main loop for running the experiment, processing each frame of the live stream.
        """
        print(f'\nSTARTING MAIN LOOP')

        # Initialize vars for tracking and depth
        prev_frames = None
        curr_frames = None
        depth_img = None
        fpss = []
        outputs = []
        prev_outputs = np.array([])

        self.ready_for_next_trial = True
        self.target_entered = True  # wait for press of "s" button to start first trial
        self.class_target_obj = -1  # placeholder value not assigned to any specific object
        self.orig_classes_obj = self.classes_obj

        grasped = False
        vibration_timer = None

        # Start key listener thread
        key_queue = queue.Queue()  # Fresh queue for each trial
        pressed_key = None
        listener_thread = threading.Thread(target=key_listener, args=(key_queue,), daemon=True)
        listener_thread.start()

        # Data processing: Iterate over each frame of the live stream
        for frame, (path, im, im0s, vid_cap, _) in enumerate(self.dataset):

            # Start timer for FPS measure
            start = time.perf_counter()

            # Setup saving and visualization
            p, im0 = Path(path[0]), im0s[0].copy()  # idx 0 is for first source (and we only have one source)
            save_path = str(save_dir / p.name)  # im.jpg
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names_obj))

            # Image pre-processing
            with self.dt[0]:
                image = torch.from_numpy(im).to(self.model_obj.device)
                image = image.half() if self.model_hand.fp16 else image.float()
                image /= 255  # 0 - 255 to 0.0 - 1.0
                if len(image.shape) == 3:
                    image = image[None]  # expand for batch dim

            # Object detection inference
            with self.dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.visualize else False
                pred_target = self.model_obj(image, augment=self.augment, visualize=visualize)
                pred_hand = self.model_hand(image, augment=self.augment, visualize=visualize)

            # Non-maximal suppression
            with self.dt[2]:
                pred_target = non_max_suppression(pred_target, self.conf_thres, self.iou_thres, self.classes_obj, self.agnostic_nms, max_det=self.max_det)
                pred_hand = non_max_suppression(pred_hand, self.conf_thres, self.iou_thres, self.classes_hand, self.agnostic_nms, max_det=self.max_det)

            for hand in pred_hand[0]:
                if len(hand):
                    hand[5] += index_add  # assign correct classID by adding len(coco_labels)

            # Camera motion compensation for tracker (ECC)
            curr_frames = im0
            if self.run_object_tracker:
                self.tracker.tracker.camera_update(prev_frames, curr_frames)

            # Initialize/clear detections
            xywhs = torch.empty(0, 4)
            confs = torch.empty(0)
            clss = torch.empty(0)

            # Process object detections
            preds = torch.cat((pred_target[0], pred_hand[0]), dim=0)  # (x, y, x, y, conf, cls)
            if len(preds) > 0:
                preds[:, :4] = scale_boxes(im.shape[2:], preds[:, :4], im0.shape).round()
                xywhs = xyxy2xywh(preds[:, :4])
                confs = preds[:, 4]
                clss = preds[:, 5]

            # Generate tracker outputs for navigation
            if self.run_object_tracker:
                # Update previous information
                outputs = self.tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)  # (x, y, x, y, track_id, cls, conf)

                # Kill tracks of old objects right upon starting the trial
                if not self.ready_for_next_trial:
                    hand_index_list = [hand + index_add for hand in self.classes_hand]
                    outputs = [output for output in outputs if output[5] in self.classes_obj + hand_index_list]

            else:  # without tracking
                outputs = np.array(preds.cpu())  # (x, y, x, y, conf, cls)
                outputs = np.insert(outputs, 4, -1, axis=1)  # insert track_id placeholder --> (x, y, x, y, track_id, conf, cls)
                outputs[:, [5, 6]] = outputs[:, [6, 5]]  # switch cls and conf to match tracker --> (x, y, x, y, track_id, cls, conf)

            # Convert xyxy to xywh
            outputs = [np.concatenate((xyxy2xywh(bb[:4]), bb[4:])) for bb in outputs]  # (x, y, w, h, track_id, cls, conf)

            # Add depth placeholder to outputs
            outputs = [np.append(bb, -1) for bb in outputs]  # (x, y, w, h, track_id, cls, conf, depth)

            # Calculate difference between current and previous frame
            if prev_frames is not None:
                img_gr_1, img_gr_2 = cv2.cvtColor(curr_frames, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_frames, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(img_gr_1, img_gr_2)
                mean_diff = np.mean(diff)
                if mean_diff > 30:  # Big change between frames
                    outputs = []

            # Depth estimation
            if not self.run_depth_estimator:
                depth_img = None
            else:
                depth_img, outputs = self.get_depth(im0, frame, outputs, prev_outputs, depthmap=depth_img, frame_factor=10)

            # Set current tracking information as previous info
            prev_outputs = np.array(outputs) if len(outputs) > 0 else np.empty((0, 8))

            # Get FPS
            end = time.perf_counter()
            runtime = max(end - start, 1e-6)
            fps = 1.0 / runtime
            fpss.append(fps)
            prev_frames = curr_frames

            # Get the target object class
            if not self.target_entered:
                if self.manual_entry:
                    user_in = "n"
                    while user_in == "n":
                        print("These are the available objects:")
                        print(coco_labels)
                        target_obj_verb = input('Enter the object you want to target: ')

                        if target_obj_verb in coco_labels.values():
                            user_in = input("Selected object is " + target_obj_verb + ". Correct? [y,n]")
                            self.class_target_obj = next(key for key, value in coco_labels.items() if value == target_obj_verb)
                            file = f'resources/sound/{target_obj_verb}.mp3'
                            # playsound(str(file))
                        else:
                            print(f'The object {target_obj_verb} is not in the list of available targets. Please reselect.')
                else:
                    target_obj_verb = self.target_objs[self.obj_index]
                    self.class_target_obj = next(key for key, value in coco_labels.items() if value == target_obj_verb)
                    file = f'resources/sound/{target_obj_verb}.mp3'
                    self.output_data.append(self.class_target_obj)
                    # playsound(str(file))

                self.target_entered = True
                self.classes_obj = [self.class_target_obj]  # only detect the target object
                grasped = False
                vibration_timer = None

            # Navigate the hand based on information from last frame and current frame detections
            if not grasped:
                grasped, curr_target = self.bracelet_controller.navigate_hand(
                    self.belt_controller,
                    outputs,
                    self.class_target_obj,
                    [hand + index_add for hand in self.classes_hand],
                    depth_img,
                    self.participant_vibration_intensities,
                    self.metric
                )
            else:  # if grasping signal was sent stop navigation process and reset target
                if vibration_timer is None:
                    vibration_timer = time.time()
                    grasped, curr_target = True, None
                elif vibration_timer > 0:
                    if time.time() - vibration_timer > 1.5:
                        if self.belt_controller:
                            self.belt_controller.stop_vibration()
                        vibration_timer = -1

            # VISUALIZATIONS

            # Write results
            for *xywh, obj_id, cls, conf, depth in outputs:
                id, cls = int(obj_id), int(cls)
                xyxy = xywh2xyxy(np.array(xywh))
                if save_img or self.save_crop or self.view_img:
                    label = None if self.hide_labels else (
                        f'ID: {id} {self.master_label[cls]}' if self.hide_conf else (f'ID: {id} {self.master_label[cls]} {conf:.2f} {depth:.2f}')
                    )
                    annotator.box_label(xyxy, label, color=colors(cls, True))

            # Target BB
            if curr_target is not None:
                for *xywh, obj_id, cls, conf, depth in [curr_target]:
                    xyxy = xywh2xyxy(np.array(xywh))
                    if save_img or self.save_crop or self.view_img:
                        label = None if self.hide_labels else 'Target object'
                        annotator.box_label(xyxy, label, color=(0, 0, 0))

            # Display results
            im0 = annotator.result()
            if self.view_img:
                cv2.putText(im0, f'FPS: {int(fps)}, Avg: {int(np.mean(fpss))}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
                if self.run_depth_estimator and depth_img is not None:
                    side_by_side = create_side_by_side(im0, depth_img, False)  # original image & depth side-by-side
                    cv2.imshow("AIBox & Depth", side_by_side)
                else:
                    cv2.imshow("AIBox", im0)
                    cv2.setWindowProperty("AIBox", cv2.WND_PROP_TOPMOST, 1)

                pressed_key = cv2.waitKey(1)
                trial_info = self.experiment_trial_logic(pressed_key)

                if trial_info == "break":
                    break

            # Save results
            if save_img:
                if self.dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[0] != save_path:  # new video
                        vid_path[0] = save_path
                        if isinstance(vid_writer[0], cv2.VideoWriter):
                            vid_writer[0].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 8, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[0].write(im0)

    @smart_inference_mode()
    def run(self):

        # Experiment setup
        if not self.manual_entry:
            target_objs = self.target_objs
            self.obj_index = 0
            print(f'The experiment will be run automatically. The selected target objects, in sequence, are:\n{target_objs}')
        else:
            print('The experiment will be run manually. You will enter the desired target for each run yourself.')

        horizontal_in, vertical_in = False, False
        self.target_entered = False
        # play_start()  # play welcome sound

        # Configure saving
        source = self.source
        save_img = not self.nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')

        if is_url and is_file:
            source = check_file(source)  # download

        save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        if save_img:
            (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load object detection models
        self.load_object_detector()

        # Load data stream
        self.bs = 1  # batch_size
        view_img = check_imshow(warn=True)

        try:
            self.dataset = LoadStreams(source, img_size=640)

        except AssertionError:
            while True:
                change_camera = input(f'Failed to open camera with index {source}. Do you want to continue with webcam? (y/n)')
                if change_camera == 'y':
                    source = '0'
                    self.dataset = LoadStreams(source, img_size=640)
                    break
                elif change_camera == 'n':
                    exit()

        self.bs = len(self.dataset)
        vid_path, vid_writer = [None] * self.bs, [None] * self.bs

        # Create combined label dictionary
        index_add = len(self.names_obj)
        labels_hand_adj = {key + index_add: value for key, value in self.names_hand.items()}
        self.master_label = self.names_obj | labels_hand_adj

        # Load tracker model
        if self.run_object_tracker:
            self.load_object_tracker(max_age=self.tracker_max_age, n_init=self.tracker_n_init)
        else:
            print('SKIPPING OBJECT TRACKER INITIALIZATION')

        # Load depth estimator
        if self.run_depth_estimator:
            self.load_depth_estimator()
        else:
            print('SKIPPING DEPTH ESTIMATOR INITIALIZATION')

        # Warmup models
        # self.warmup_model(self.model_obj)
        self.warmup_model(self.model_hand)
        if self.run_object_tracker:
            self.warmup_model(self.tracker.model, 'tracker')

        # Start experiment loop
        self.experiment_loop(save_dir, save_img, index_add, vid_path, vid_writer)