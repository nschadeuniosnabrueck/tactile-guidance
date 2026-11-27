#!/usr/bin/env python
"""
experiment.py - combined auditory / tactile grasping experiment (script v2.1, frame-gated).

Tasks always run in this order: grasping -> multiple_objects -> depth_navigation.
Within each task, auditory and tactile blocks alternate (A T A T ... or T A T A ...).

Keys, in the OpenCV window:
  READY state
    S        start trial
    O        set obstacle/layout label in the terminal (kept until changed)
    C        abort session
  RUNNING state
    Y / N / F / T   end trial: success / fail / system fail / wrong target
    C               abort session (current trial logged as 'aborted')
    Auditory only: arrow keys = L/R/U/D instruction, G = grasp instruction
  BREAK screen
    SPACE    continue,  C  abort

Tactile condition: full pipeline + bracelet guidance. Perception and bracelet logic run
only on NEW camera frames; duplicate frames reuse the last result.
Auditory condition: camera preview only, no models run, experimenter guides verbally.

Success without a grasp signal (tactile): the experimenter is asked in the terminal
  success_no_signal = 1  -> kept as success
  success_no_signal = 0  -> rejected, stored as outcome 'success_no_signal_rejected' (key n)
  NA                     -> not applicable

Outputs (results/experiment/):
  participant_<p>_<stamp>_trials.csv   one row per trial
  participant_<p>_<stamp>_frames.csv   one row per new camera frame during trials
  participant_<p>_<stamp>_config.json  settings, targets, calibration, plan, file hashes
  videos/participant_<p>_<stamp>/      per-block videos + *_timestamps.csv (optional)
"""

import sys
import csv
import json
import math
import time
import types
import random
import argparse
import hashlib
import threading
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parent
for sub in ("yolov5", "strongsort", "midas", "unidepth"):
    sys.path.append(str(root / sub))

import cv2
import numpy as np
import torch

SCRIPT_VERSION = "2.1-frame-gated"
MOVE_BACK_MIN_S = 0.5          # move-back shorter than this is treated as flicker


# xformers shim: UniDepth (V1 and V2 DepthHead) imports
# xformers.components.attention.NystromAttention. Pure-PyTorch port, no params.
def _ensure_xformers_nystrom():
    try:
        from xformers.components.attention import NystromAttention  # noqa: F401
        return
    except Exception:
        pass

    def _softmax_qk(q, k):
        q = q / math.sqrt(k.size(-1))
        return torch.softmax(q @ k.transpose(-2, -1), dim=-1)

    def _avg_pool_landmarks(x, n):
        seq_len, d = x.shape[-2], x.shape[-1]
        segments = math.ceil(seq_len / n)
        pad = segments * n - seq_len
        if pad:
            x = torch.cat([x, x.new_zeros(*x.shape[:-2], pad, d)], dim=-2)
        return x.reshape(*x.shape[:-2], n, segments, d).mean(dim=-2)

    def _iterative_pinv(k, n_iter=6):
        eye = torch.eye(k.size(-1), device=k.device, dtype=k.dtype)
        col_max = torch.sum(k, dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
        v = k.transpose(-1, -2) / col_max
        for _ in range(n_iter):
            kv = k @ v
            v = (0.25 * v) @ (13 * eye - kv @ (15 * eye - kv @ (7 * eye - kv)))
        return v

    class NystromAttention(torch.nn.Module):
        def __init__(self, dropout=0.0, num_heads=1, num_landmarks=64,
                     landmark_pooling=None, causal=False, use_razavi_pinverse=True,
                     pinverse_original_init=False, inv_iterations=6,
                     v_skip_connection=None, conv_kernel_size=None, *args, **kwargs):
            super().__init__()
            if conv_kernel_size is not None or v_skip_connection is not None or causal:
                raise NotImplementedError("Shim supports only the UniDepth configuration.")
            self.num_heads = num_heads
            self.num_landmarks = num_landmarks
            self.use_razavi_pinverse = use_razavi_pinverse
            self.inv_iterations = inv_iterations
            self.attn_drop = torch.nn.Dropout(dropout)

        def forward(self, q, k, v, key_padding_mask=None, *args, **kwargs):
            if key_padding_mask is not None:
                raise NotImplementedError("key_padding_mask not supported by shim.")
            if self.num_landmarks >= k.size(-2):
                x = _softmax_qk(q, k) @ v
            else:
                q_l = _avg_pool_landmarks(q, self.num_landmarks)
                k_l = _avg_pool_landmarks(k, self.num_landmarks)
                k1 = _softmax_qk(q, k_l)
                k2 = _softmax_qk(q_l, k_l)
                k3 = _softmax_qk(q_l, k) @ v
                k2_inv = (_iterative_pinv(k2, self.inv_iterations)
                          if self.use_razavi_pinverse else torch.linalg.pinv(k2))
                x = (k1 @ k2_inv) @ k3
            return self.attn_drop(x)

    for name in ("xformers", "xformers.components", "xformers.components.attention"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
    sys.modules["xformers"].components = sys.modules["xformers.components"]
    sys.modules["xformers.components"].attention = sys.modules["xformers.components.attention"]
    sys.modules["xformers.components.attention"].NystromAttention = NystromAttention
    print("Using pure-PyTorch NystromAttention shim (xformers not available).")


_ensure_xformers_nystrom()

import controller                                   # also loads model libs / patches  # noqa: E402,F401
from controller import TaskController               # noqa: E402
from labels import coco_labels                      # noqa: E402
from bracelet import connect_belt, BraceletController  # noqa: E402
from midas.run import create_side_by_side           # noqa: E402
from yolov5.utils.augmentations import letterbox    # noqa: E402
from yolov5.utils.general import non_max_suppression, scale_boxes, xyxy2xywh, xywh2xyxy  # noqa: E402
from yolov5.utils.plots import Annotator, colors    # noqa: E402
from yolov5.utils.torch_utils import smart_inference_mode  # noqa: E402


# Constants
TASKS = ["grasping", "multiple_objects", "depth_navigation"]
TASK_TARGETS = {
    "grasping": ["bottle", "clock", "potted plant", "bowl", "cup"],
    "multiple_objects": ["bottle"],
    "depth_navigation": ["bottle"],
}
OUTCOMES = {"y": "success", "n": "fail", "f": "system_fail", "t": "wrong_target"}
WINDOW = "Experiment"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

ARROWS = {
    2424832: "left", 65361: "left", 63234: "left",
    2490368: "up", 65362: "up", 63232: "up",
    2555904: "right", 65363: "right", 63235: "right",
    2621440: "down", 65364: "down", 63233: "down",
}
INSTR_SYMBOL = {"left": "L", "right": "R", "up": "U", "down": "D", "g": "G"}

FIELDS = [
    "participant", "task", "condition", "first_condition", "block", "n_blocks", "block_global",
    "trial_in_block", "trial_in_condition", "target", "obstacle_layout",
    "start_time", "end_time", "duration_s",
    "navigation_onset_s", "freezing_onset_s", "grasp_signal_s", "grasp_sent", "success_no_signal",
    "first_instr_s", "grasp_instr_s", "num_instructions", "instructions", "instr_times_s",
    "n_frames", "mean_fps", "hand_detect_rate", "target_detect_rate",
    "obstacle_frame_rate", "first_obstacle_s", "n_move_back", "move_back_s",
    "video_raw", "video_annotated",
    "outcome_key", "outcome",
]

FRAME_FIELDS = [
    "participant", "task", "condition", "block_global", "trial_in_condition", "target",
    "frame", "camera_frame", "t_rel_s", "wall_time", "fps", "raw_video_frame", "annotated_video_frame",
    "n_dets",
    "hand_x", "hand_y", "hand_w", "hand_h", "hand_conf", "hand_depth",
    "target_x", "target_y", "target_w", "target_h", "target_conf", "target_depth",
    "hand_detected", "target_detected",
    "frozen", "grasped", "obstacle", "obstacle_x", "obstacle_y", "roi_min_y", "move_back",
    "int_right", "int_left", "int_top", "int_bottom",
    "instruction",
    "t_perceive_ms", "t_nav_ms",
]


def decode_key(k):
    if k == -1:
        return None
    if k in ARROWS:
        return ARROWS[k]
    if (k & 0xFFFF) in ARROWS:
        return ARROWS[k & 0xFFFF]
    k8 = k & 0xFF
    if 32 <= k8 < 127:
        return chr(k8).lower()
    return None


def draw_text(img, lines, color=(0, 255, 0), origin=(15, 30), scale=0.7, step=30):
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x, y), FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), FONT, scale, color, 2, cv2.LINE_AA)
        y += step
    return img


def bb_values(bb):
    """(x, y, w, h, conf, depth) of a detection, or NA."""
    if bb is None:
        return ["NA"] * 6
    try:
        b = np.asarray(bb, dtype=float).ravel()
        depth = round(float(b[7]), 3) if len(b) > 7 else "NA"
        return [round(float(b[i]), 1) for i in range(4)] + [round(float(b[6]), 3), depth]
    except Exception:
        return ["NA"] * 6


def point_xy(p):
    try:
        a = np.asarray(p, dtype=float).ravel()
        return round(float(a[0]), 1), round(float(a[1]), 1)
    except Exception:
        return "NA", "NA"


def write_timestamps(path, times):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_frame", "wall_time"])
        for i, t in enumerate(times):
            w.writerow([i, round(t, 4)])


# Camera: shared threaded capture, frame counter, optional raw recording
class Camera:
    def __init__(self, source):
        src = int(source) if str(source).isnumeric() else source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera source {source}")
        ok, frame = self.cap.read()
        if not ok:
            raise IOError(f"Camera source {source} returned no frame")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 <= fps <= 120 else 30.0
        self.frame = frame
        self.seq = 0                  # increments on every new camera frame
        self.lock = threading.Lock()
        self.writer = None
        self.rec_path = None
        self.rec_times = []
        self.rec_count = 0
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    @property
    def recording(self):
        return self.writer is not None

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.seq += 1
                if self.writer is not None:
                    self.writer.write(frame)
                    self.rec_times.append(time.time())
                    self.rec_count += 1

    def read(self):
        """Returns (frame copy, raw video index or 'NA', camera sequence number)."""
        with self.lock:
            idx = self.rec_count - 1 if self.writer is not None else "NA"
            return self.frame.copy(), idx, self.seq

    def start_recording(self, path):
        h, w = self.frame.shape[:2]
        wr = cv2.VideoWriter(str(path), FOURCC, self.fps, (w, h))
        with self.lock:
            self.rec_times, self.rec_count = [], 0
            self.rec_path = Path(path)
            self.writer = wr

    def stop_recording(self):
        with self.lock:
            wr, self.writer = self.writer, None
            times, path = self.rec_times, self.rec_path
        if wr is not None:
            wr.release()
            write_timestamps(path.with_name(path.stem + "_timestamps.csv"), times)

    def release(self):
        self.stop_recording()
        self.running = False
        time.sleep(0.05)
        self.cap.release()


def open_camera(source):
    while True:
        try:
            return Camera(source)
        except IOError as e:
            print(e)
            ans = input("Continue with webcam (source 0)? (y/n) ").strip().lower()
            if ans == "y":
                source = "0"
            elif ans == "n":
                sys.exit()


# Experiment controller
class ExperimentController(TaskController):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name2id = {v: k for k, v in coco_labels.items()}
        self.bracelet_controller = None
        self.run_object_tracker = False
        self.run_depth_estimator = False
        self.camera = None
        self.obstacle_layout = ""
        self.ann_writer = None
        self.ann_path = None
        self.ann_size = None
        self.ann_times = []
        self.raw_path = None
        self.frame_file = None
        self.frame_writer = None
        self._nav_this_frame = False
        self._last_timing = {"t_perceive_ms": "NA", "t_nav_ms": "NA"}

    # Setup
    def setup(self):
        self.bs = 1
        self.load_object_detector()
        self.stride = int(self.model_obj.stride)
        self.pt_obj = self.model_obj.pt
        self.index_add = len(self.names_obj)
        labels_hand = {k + self.index_add: v for k, v in self.names_hand.items()}
        self.master_label = dict(self.names_obj) | labels_hand
        self.hand_ids = [h + self.index_add for h in self.classes_hand]

        self.load_object_tracker(max_age=self.tracker_max_age, n_init=self.tracker_n_init)
        self.load_depth_estimator()

        print("\nWARMING UP MODELS...")
        for m in (self.model_obj, self.model_hand):
            try:
                m.warmup(imgsz=(1, 3, *self.imgsz))
            except Exception as e:
                print(f"Warmup skipped: {e}")
        try:
            self.tracker.model.warmup()
        except Exception as e:
            print(f"Tracker warmup skipped: {e}")

        self.camera = open_camera(self.source)

        print("WARMING UP DEPTH ESTIMATOR...")
        for _ in range(2):
            self._predict_and_normalize_depth(self.camera.read()[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.0f} MB, "
                  f"reserved: {torch.cuda.memory_reserved() / 1e6:.0f} MB")

        if self.save_frames:
            self.frame_file = open(self.frames_path, "w", newline="")
            self.frame_writer = csv.DictWriter(self.frame_file, fieldnames=FRAME_FIELDS)
            self.frame_writer.writeheader()

        if self.save_video != "none":
            self.video_dir.mkdir(parents=True, exist_ok=True)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    def configure_task(self, task):
        self.task = task
        self.run_object_tracker = task == "multiple_objects"
        self.run_depth_estimator = task == "depth_navigation"

    def reset_perception(self):
        self.prev_frame = None
        self.prev_outputs = np.empty((0, 8))
        self.depth_img = None
        trk = getattr(getattr(self, "tracker", None), "tracker", None)
        if trk is not None:
            if hasattr(trk, "tracks"):
                trk.tracks = []
            if hasattr(trk, "_next_id"):
                trk._next_id = 1

    # Video recording
    def start_block_video(self, blk):
        base = f"{self.task}_{blk['condition']}_B{blk['block']}_G{blk['block_global']:02d}"
        self.raw_path = self.ann_path = None
        if self.save_video in ("raw", "both"):
            self.raw_path = self.video_dir / f"{base}_raw.mp4"
            self.camera.start_recording(self.raw_path)
        if self.save_video in ("annotated", "both"):
            self.ann_path = self.video_dir / f"{base}_annotated.mp4"
            self.ann_writer, self.ann_size, self.ann_times = None, None, []

    def write_annotated(self, img):
        """Returns the index of this frame in the annotated video, or 'NA'."""
        if self.ann_path is None:
            return "NA"
        if self.ann_writer is None:
            self.ann_size = (img.shape[1], img.shape[0])
            self.ann_writer = cv2.VideoWriter(str(self.ann_path), FOURCC,
                                              self.video_fps, self.ann_size)
        if (img.shape[1], img.shape[0]) != self.ann_size:
            img = cv2.resize(img, self.ann_size)
        self.ann_writer.write(img)
        self.ann_times.append(time.time())
        return len(self.ann_times) - 1

    def stop_block_video(self):
        if self.camera is not None:
            self.camera.stop_recording()
        if self.ann_writer is not None:
            self.ann_writer.release()
            write_timestamps(self.ann_path.with_name(self.ann_path.stem + "_timestamps.csv"),
                             self.ann_times)
        self.ann_writer = None

    # Perception: object detection, hand detection, tracking, depth estimation
    @smart_inference_mode()
    def perceive(self, im0, frame_idx, classes_obj, running):
        im = letterbox(im0, 640, stride=self.stride, auto=self.pt_obj)[0]
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])[None]
        image = torch.from_numpy(im).to(self.model_obj.device)
        image = image.half() if self.model_obj.fp16 else image.float()
        image /= 255

        pred_t = non_max_suppression(self.model_obj(image, augment=self.augment), self.conf_thres,
                                     self.iou_thres, classes_obj, self.agnostic_nms, max_det=self.max_det)
        pred_h = non_max_suppression(self.model_hand(image, augment=self.augment), self.conf_thres,
                                     self.iou_thres, self.classes_hand, self.agnostic_nms, max_det=self.max_det)
        hands = pred_h[0]
        if len(hands):
            hands[:, 5] += self.index_add
        preds = torch.cat((pred_t[0], hands), dim=0)

        if self.run_object_tracker:
            self.tracker.tracker.camera_update(self.prev_frame, im0)

        xywhs, confs, clss = torch.empty(0, 4), torch.empty(0), torch.empty(0)
        if len(preds):
            preds[:, :4] = scale_boxes(im.shape[2:], preds[:, :4], im0.shape).round()
            xywhs, confs, clss = xyxy2xywh(preds[:, :4]), preds[:, 4], preds[:, 5]

        if self.run_object_tracker:
            outputs = self.tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
            if running:
                keep = list(classes_obj) + self.hand_ids
                outputs = [o for o in outputs if o[5] in keep]
        else:
            outputs = preds.cpu().numpy()
            outputs = np.insert(outputs, 4, -1, axis=1)
            outputs[:, [5, 6]] = outputs[:, [6, 5]]

        outputs = [np.concatenate((xyxy2xywh(bb[:4]), bb[4:])) for bb in outputs]
        outputs = [np.append(bb, -1) for bb in outputs]

        if self.prev_frame is not None and self.prev_frame.shape == im0.shape:
            diff = cv2.absdiff(cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY))
            if np.mean(diff) > 30:
                outputs = []

        depth_img = None
        if self.run_depth_estimator:
            self.depth_img, outputs = self.get_depth(im0, frame_idx, outputs, self.prev_outputs,
                                                     depthmap=self.depth_img, frame_factor=10)
            depth_img = self.depth_img

        self.prev_outputs = np.array(outputs) if len(outputs) else np.empty((0, 8))
        self.prev_frame = im0
        return outputs, depth_img

    def navigation_step(self, rec, outputs, target_cls, depth_img):
        bc = self.bracelet_controller
        self._nav_this_frame = False
        if not rec["grasped"]:
            self._nav_this_frame = True
            grasped, curr_target = bc.navigate_hand(
                self.belt_controller, outputs, target_cls, self.hand_ids, depth_img,
                self.participant_vibration_intensities, self.metric)
            rec["grasped"] = grasped
            if rec["freeze_t"] is None and bc.freezing_time != "NA":
                rec["freeze_t"] = bc.freezing_time
            return curr_target
        if rec["vib_timer"] is None:
            rec["vib_timer"] = time.time()
        elif rec["vib_timer"] > 0 and time.time() - rec["vib_timer"] > 1.5:
            self.stop_belt()
            rec["vib_timer"] = -1
        return None

    # Move-back detection: obstacle in front of hand, hand and target both present, ROI <= 5
    def _close_move_back(self, rec, t_rel, min_s=MOVE_BACK_MIN_S):
        """End an open move-back episode. Episodes shorter than min_s are treated as flicker."""
        if rec["mb_start"] is None:
            return
        dur = t_rel - rec["mb_start"]
        rec["mb_start"] = None
        if dur >= min_s:
            rec["mb_episodes"] += 1
            rec["mb_time_s"] += dur

    def _update_move_back(self, rec, mb, t_rel):
        if mb and rec["mb_start"] is None:
            rec["mb_start"] = t_rel
        elif not mb:
            self._close_move_back(rec, t_rel)

    # Logging: one row per new camera frame, with perception and bracelet state
    def log_frame(self, rec, blk, i_cond, target, target_cls, outputs, fps,
                  raw_idx, cam_seq, ann_idx, instr, tactile):
        now = time.time()
        t_rel = now - rec["start"]
        rec["frames"] += 1
        if fps is not None:
            rec["fps"].append(fps)

        row = dict.fromkeys(FRAME_FIELDS, "NA")
        row.update(participant=self.participant, task=self.task, condition=blk["condition"],
                   block_global=blk["block_global"], trial_in_condition=i_cond, target=target,
                   frame=rec["frames"], camera_frame=cam_seq, t_rel_s=round(t_rel, 3),
                   wall_time=round(now, 4),
                   fps=round(fps, 1) if fps is not None else "NA",
                   raw_video_frame=raw_idx, annotated_video_frame=ann_idx,
                   instruction=instr, n_dets=len(outputs))
        row.update(self._last_timing)

        if tactile:
            bc = self.bracelet_controller
            has_hand = any(o[5] in self.hand_ids for o in outputs)
            has_tgt = any(int(o[5]) == target_cls for o in outputs)
            rec["hand_frames"] += has_hand
            rec["target_frames"] += has_tgt

            nav = self._nav_this_frame
            hand = bc.prev_hand if nav else None
            tgt = bc.prev_target if nav else None
            obs = bc.obstacle_target if nav else None
            is_obs = obs is not None
            roi = bc.roi_min_y
            mb = bool(nav and is_obs and roi <= 5 and hand is not None and tgt is not None)

            if is_obs:
                rec["obstacle_frames"] += 1
                if rec["first_obstacle"] is None:
                    rec["first_obstacle"] = round(t_rel, 3)
            self._update_move_back(rec, mb, t_rel)

            ints = ["NA"] * 4
            if nav and hand is not None and tgt is not None and not mb and not rec["grasped"]:
                guide = obs if is_obs else tgt
                try:
                    r, l, t, b, _ = bc.get_intensity(hand, guide, self.participant_vibration_intensities)
                    ints = [r, l, t, b]
                except Exception:
                    pass

            hx = bb_values(hand)
            tx = bb_values(tgt)
            ox, oy = point_xy(obs) if is_obs else ("NA", "NA")
            row.update(
                hand_x=hx[0], hand_y=hx[1], hand_w=hx[2], hand_h=hx[3], hand_conf=hx[4], hand_depth=hx[5],
                target_x=tx[0], target_y=tx[1], target_w=tx[2], target_h=tx[3],
                target_conf=tx[4], target_depth=tx[5],
                hand_detected=int(has_hand), target_detected=int(has_tgt),
                frozen=int(bool(bc.frozen)), grasped=int(bool(rec["grasped"])),
                obstacle=int(is_obs), obstacle_x=ox, obstacle_y=oy,
                roi_min_y=round(float(roi), 1) if is_obs else "NA",
                move_back=int(mb),
                int_right=ints[0], int_left=ints[1], int_top=ints[2], int_bottom=ints[3],
            )

        if self.frame_writer is not None:
            self.frame_writer.writerow(row)

    # Helper functions
    def stop_belt(self):
        if self.belt_controller:
            try:
                self.belt_controller.stop_vibration()
            except Exception:
                pass

    def write_row(self, row):
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)

    def render(self, im0, outputs, curr_target, depth_img, hud, color):
        if len(outputs) or curr_target is not None:
            ann = Annotator(im0, line_width=self.line_thickness, example=str(self.names_obj))
            for *xywh, obj_id, cls, conf, depth in outputs:
                cls = int(cls)
                label = f"ID: {int(obj_id)} {self.master_label.get(cls, cls)} {conf:.2f} {depth:.2f}"
                ann.box_label(xywh2xyxy(np.array(xywh)), label, color=colors(cls, True))
            if curr_target is not None:
                ann.box_label(xywh2xyxy(np.array(curr_target[:4])), "Target object", color=(0, 0, 0))
            im0 = ann.result()
        bc = self.bracelet_controller
        if bc is not None and getattr(bc, "obstacle_target", None) is not None:
            ox, oy = point_xy(bc.obstacle_target)
            if ox != "NA":
                cv2.drawMarker(im0, (int(ox), int(oy)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 25, 3)
        draw_text(im0, hud, color)
        if depth_img is not None:
            im0 = create_side_by_side(im0, depth_img, False)
        return im0

    def wait_screen(self, lines, color=(0, 255, 255)):
        print("\n" + "\n".join(lines))
        t0 = time.time()
        while True:
            frame = (self.camera.read()[0] * 0.3).astype(np.uint8)
            draw_text(frame, lines + ["", f"break: {time.time() - t0:5.0f} s"],
                      color, origin=(30, 60), scale=0.9, step=40)
            cv2.imshow(WINDOW, frame)
            k = decode_key(cv2.waitKeyEx(30))
            if k == " ":
                print(f"(continued after {time.time() - t0:.0f} s)")
                return True
            if k == "c":
                return False

    def ask_layout(self, where):
        print(f"\nObstacle layout for {where}.")
        cur = self.obstacle_layout or "none"
        lbl = input(f"  current: '{cur}' | new label, Enter = keep, '-' = clear: ").strip()
        if lbl == "-":
            self.obstacle_layout = ""
        elif lbl:
            self.obstacle_layout = lbl
        print(f"  layout now: '{self.obstacle_layout or 'none'}'")

    # Trials logic
    def start_trial(self, cond):
        rec = dict(start=time.time(), grasped=False, vib_timer=None, freeze_t=None,
                   n_instr=0, instr="", last_instr=None, instr_times=[],
                   first_instr=None, grasp_instr=None,
                   frames=0, fps=[], hand_frames=0, target_frames=0,
                   obstacle_frames=0, first_obstacle=None,
                   mb_start=None, mb_episodes=0, mb_time_s=0.0)
        if cond == "tactile":
            self.bracelet_controller = BraceletController(
                vibration_intensities=self.participant_vibration_intensities)
            self.bracelet_controller.navigation_time = None
            self.bracelet_controller.vibrate = True
        return rec

    def finish_trial(self, rec, blk, i_blk, i_cond, target, key, outcome):
        end = time.time()
        self._close_move_back(rec, end - rec["start"])
        self.stop_belt()
        aud = blk["condition"] == "auditory"
        rel = lambda t: round(t - rec["start"], 3) if isinstance(t, (int, float)) else "NA"
        nav = frz = grs = "NA"
        if not aud and self.bracelet_controller:
            bc = self.bracelet_controller
            nav, frz, grs = rel(bc.navigation_time), rel(rec["freeze_t"]), rel(bc.grasping_time)

        # Success without grasp signal: experimenter decides now and it is recorded.
        success_no_signal = "NA"
        if not aud and outcome == "success" and grs == "NA":
            print("  SUCCESS without grasp signal - decide now.")
            ans = input("  Keep as success? (y = keep / n = mark as failure): ").strip().lower()
            if ans == "n":
                outcome, key = "success_no_signal_rejected", "n"
                success_no_signal = 0
            else:
                success_no_signal = 1
            print(f"  -> recorded: {outcome}")

        n = rec["frames"]
        rate = lambda c: round(c / n, 3) if (n and not aud) else "NA"
        depth_task = (not aud) and self.task == "depth_navigation"

        self.write_row(dict(
            participant=self.participant, task=self.task, condition=blk["condition"],
            first_condition=blk["first_condition"], block=blk["block"], n_blocks=blk["n_blocks"],
            block_global=blk["block_global"], trial_in_block=i_blk, trial_in_condition=i_cond,
            target=target, obstacle_layout=self.obstacle_layout or "NA",
            start_time=round(rec["start"], 3), end_time=round(end, 3),
            duration_s=round(end - rec["start"], 3),
            navigation_onset_s=nav, freezing_onset_s=frz, grasp_signal_s=grs,
            grasp_sent=("NA" if aud else int(grs != "NA")),
            success_no_signal=success_no_signal,
            first_instr_s=rec["first_instr"] if aud and rec["first_instr"] is not None else "NA",
            grasp_instr_s=rec["grasp_instr"] if aud and rec["grasp_instr"] is not None else "NA",
            num_instructions=rec["n_instr"] if aud else "NA",
            instructions=rec["instr"] if aud else "NA",
            instr_times_s=";".join(map(str, rec["instr_times"])) if aud else "NA",
            n_frames=n, mean_fps=round(float(np.mean(rec["fps"])), 1) if rec["fps"] else "NA",
            hand_detect_rate=rate(rec["hand_frames"]),
            target_detect_rate=rate(rec["target_frames"]),
            obstacle_frame_rate=rate(rec["obstacle_frames"]) if depth_task else "NA",
            first_obstacle_s=(rec["first_obstacle"] if depth_task and rec["first_obstacle"] is not None
                              else "NA"),
            n_move_back=rec["mb_episodes"] if depth_task else "NA",
            move_back_s=round(rec["mb_time_s"], 3) if depth_task else "NA",
            video_raw=self.raw_path.name if self.raw_path else "NA",
            video_annotated=self.ann_path.name if self.ann_path else "NA",
            outcome_key=key, outcome=outcome))
        if self.frame_file:
            self.frame_file.flush()
        print(f"  -> {outcome.upper()} ({end - rec['start']:.2f} s, {n} frames)")
        self.bracelet_controller = None

    def run_block(self, blk):
        self.start_block_video(blk)
        try:
            return self._run_block(blk)
        finally:
            self.stop_block_video()
            self.stop_belt()

    def _run_block(self, blk):
        cond = blk["condition"]
        tactile = cond == "tactile"
        color = (0, 255, 0) if tactile else (0, 165, 255)
        self.reset_perception()
        self.stop_belt()
        trials = blk["trials"]
        i, state, rec, frame_idx = 0, "ready", None, 0
        last_seq, last_new_t, new_hist = None, None, []
        last_outputs, last_depth, last_curr = [], None, None

        while i < len(trials):
            i_cond, target = trials[i]
            target_cls = self.name2id[target]
            im0, raw_idx, seq = self.camera.read()
            new_frame = seq != last_seq          # True only when the camera delivered a new image
            last_seq = seq
            outputs, depth_img, curr_target = [], None, None
            self._nav_this_frame = False
            self._last_timing = {"t_perceive_ms": "NA", "t_nav_ms": "NA"}

            if tactile:
                if new_frame:
                    classes = [target_cls] if state == "running" else self.classes_obj
                    t0 = time.perf_counter()
                    outputs, depth_img = self.perceive(im0, frame_idx, classes, state == "running")
                    t1 = time.perf_counter()
                    self._last_timing["t_perceive_ms"] = round((t1 - t0) * 1000, 1)
                    if state == "running":
                        curr_target = self.navigation_step(rec, outputs, target_cls, depth_img)
                        self._last_timing["t_nav_ms"] = round((time.perf_counter() - t1) * 1000, 1)
                    last_outputs, last_depth, last_curr = outputs, depth_img, curr_target
                    frame_idx += 1
                else:
                    # duplicate camera frame: reuse last result, do not re-run models or bracelet logic
                    outputs, depth_img, curr_target = last_outputs, last_depth, last_curr

            now = time.perf_counter()
            if new_frame:
                if last_new_t is not None:
                    new_hist = (new_hist + [now - last_new_t])[-30:]
                last_new_t = now
            frame_fps = 1.0 / float(np.mean(new_hist)) if (new_frame and new_hist) else None
            fps_txt = f"{1.0 / float(np.mean(new_hist)):.0f}" if new_hist else "--"

            hud = [f"{self.task.upper()} | {cond.upper()} | block {blk['block']}/{blk['n_blocks']} "
                   f"| trial {i + 1}/{len(trials)} (#{i_cond})",
                   f"Target: {target}" + (f" | layout: {self.obstacle_layout}" if self.obstacle_layout else "")]
            if state == "ready":
                hud.append("READY - S start trial | O set layout label")
            else:
                hud.append(f"RUNNING {time.time() - rec['start']:5.1f}s | Y ok  N fail  F sys-fail  T wrong")
                if not tactile:
                    hud.append(f"Instructions: {rec['n_instr']}  {rec['instr'][-20:]}  (arrows / G)")
                elif rec["grasped"]:
                    hud.append("GRASP SIGNAL SENT")
            rec_flag = " | REC" if self.save_video != "none" else ""
            hud.append(f"FPS {fps_txt}{rec_flag} | C = abort session")

            shown = self.render(im0, outputs, curr_target, depth_img, hud, color)
            ann_idx = self.write_annotated(shown) if new_frame else "NA"
            cv2.imshow(WINDOW, shown)

            k = decode_key(cv2.waitKeyEx(1))

            if state == "ready":
                if k == "s":
                    print(f"[{self.task}/{cond}/B{blk['block']}] trial {i + 1}: target = {target}"
                          + (f", layout = {self.obstacle_layout}" if self.obstacle_layout else ""))
                    rec = self.start_trial(cond)
                    state = "running"
                elif k == "o":
                    lbl = input(f"Obstacle/layout label (current: '{self.obstacle_layout}', "
                                f"'-' to clear, Enter to keep): ").strip()
                    if lbl == "-":
                        self.obstacle_layout = ""
                    elif lbl:
                        self.obstacle_layout = lbl
                elif k == "c":
                    return "abort"
                continue

            # running
            instr = ""
            if not tactile and k in INSTR_SYMBOL and k != rec["last_instr"]:
                t_rel = round(time.time() - rec["start"], 3)
                rec["n_instr"] += 1
                rec["instr"] += INSTR_SYMBOL[k]
                rec["instr_times"].append(t_rel)
                rec["last_instr"] = k
                instr = INSTR_SYMBOL[k]
                if rec["first_instr"] is None:
                    rec["first_instr"] = t_rel
                if k == "g" and rec["grasp_instr"] is None:
                    rec["grasp_instr"] = t_rel
                print(f"    {k} @ {t_rel}s")

            timed_out = self.time_limit > 0 and time.time() - rec["start"] > self.time_limit
            ends = k in OUTCOMES or timed_out or k == "c"
            # log each new camera frame, plus any instruction or end-of-trial key press
            if new_frame or instr or ends:
                self.log_frame(rec, blk, i_cond, target, target_cls, outputs, frame_fps,
                               raw_idx, seq, ann_idx, instr, tactile)

            if k in OUTCOMES or timed_out:
                key = k if k in OUTCOMES else ""
                self.finish_trial(rec, blk, i + 1, i_cond, target, key,
                                  OUTCOMES[k] if k in OUTCOMES else "timeout")
                i += 1
                state = "ready"
            elif k == "c":
                self.finish_trial(rec, blk, i + 1, i_cond, target, "c", "aborted")
                return "abort"

        return "done"

    # Session logic
    def run_session(self, plan):
        hints = {"auditory": "AUDITORY: no bracelet - guide verbally (log with arrows / G)",
                 "tactile": "TACTILE: bracelet guidance - make sure the bracelet is worn"}
        for t_idx, (task, blocks) in enumerate(plan):
            self.configure_task(task)
            for b_idx, blk in enumerate(blocks):
                if task == "depth_navigation":
                    self.ask_layout(f"block {blk['block']}/{blk['n_blocks']} ({blk['condition']})")
                if t_idx == 0 and b_idx == 0:
                    head = "SESSION START"
                elif b_idx == 0:
                    head = f"TASK '{plan[t_idx - 1][0]}' COMPLETE - BREAK"
                else:
                    prev = blocks[b_idx - 1]
                    head = f"BLOCK COMPLETE ({prev['condition']} {prev['block']}/{prev['n_blocks']}) - BREAK"
                lines = [head, "",
                         f"Next: task {t_idx + 1}/3 '{task}'",
                         f"      {blk['condition'].upper()} block {blk['block']}/{blk['n_blocks']}"
                         f"  ({len(blk['trials'])} trials)",
                         hints[blk["condition"]],
                         f"Obstacle layout: {self.obstacle_layout or 'none'}" if task == "depth_navigation" else "",
                         "", "SPACE = start block     C = abort session"]
                if not self.wait_screen(lines):
                    return False
                if self.run_block(blk) == "abort":
                    print("SESSION ABORTED")
                    return False
        self.wait_screen(["SESSION COMPLETE", "", f"Data: {self.csv_path.name}", "",
                          "Press SPACE to close"], (0, 255, 0))
        return True

    def close(self):
        self.stop_belt()
        try:
            self.stop_block_video()
        except Exception:
            pass
        if self.camera:
            self.camera.release()
        if self.frame_file:
            self.frame_file.close()
        cv2.destroyAllWindows()
        if self.belt_controller:
            try:
                self.belt_controller.disconnect_belt()
            except Exception:
                pass


# Experiment plan 
def build_plan(trials, block_size, orders, shuffle, rng):
    plan, g = [], 0
    for task in TASKS:
        base = TASK_TARGETS[task]
        targets = [base[i % len(base)] for i in range(trials)]
        if shuffle:
            rng.shuffle(targets)
        first = orders[task]
        second = "tactile" if first == "auditory" else "auditory"
        n_blocks = math.ceil(trials / block_size)
        blocks = []
        for b in range(n_blocks):
            idx = range(b * block_size, min(trials, (b + 1) * block_size))
            for cond in (first, second):
                g += 1
                blocks.append(dict(condition=cond, first_condition=first, block=b + 1,
                                   n_blocks=n_blocks, block_global=g,
                                   trials=[(j + 1, targets[j]) for j in idx]))
        plan.append((task, blocks))
    return plan


def ask_int(prompt, min_=1):
    while True:
        try:
            v = int(input(prompt))
            if v >= min_:
                return v
        except ValueError:
            pass
        print(f"Please enter an integer >= {min_}.")


def parse_cond(s):
    s = s.strip().lower()
    if s.startswith("a"):
        return "auditory"
    if s.startswith("t"):
        return "tactile"
    raise ValueError(s)


def get_orders(arg):
    if arg:
        toks = arg.replace(",", " ").split() if ("," in arg or " " in arg) else list(arg)
        if len(toks) != len(TASKS):
            raise SystemExit(f"--orders needs {len(TASKS)} entries, e.g. ATA")
        return {t: parse_cond(c) for t, c in zip(TASKS, toks)}
    orders = {}
    for t in TASKS:
        while True:
            try:
                orders[t] = parse_cond(input(f"First condition for '{t}' [a=auditory / t=tactile]: "))
                break
            except ValueError:
                print("Enter a or t.")
    return orders


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_calibration(participant):
    try:
        with open(f"results/calibration/calibration_participant_{participant}.json") as f:
            cal = json.load(f)
        print("Calibration intensities loaded successfully.")
        return cal, "file"
    except Exception:
        while True:
            a = input("Error loading calibration file. Continue with baseline 50 for each motor? (y/n) ")
            if a == "y":
                return {"bottom": 50, "top": 50, "left": 50, "right": 50}, "baseline_fallback"
            if a == "n":
                sys.exit("Please re-import the calibration file. Aborting.")


def file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    except Exception:
        return "NA"


def main():
    ap = argparse.ArgumentParser(description="Combined auditory/tactile experiment.")
    ap.add_argument("-p", "--participant", type=int, required=True)
    ap.add_argument("--trials", type=int, help="Trials per condition per task (e.g. 10).")
    ap.add_argument("--block_size", type=int, help="Trials per condition block (e.g. 5).")
    ap.add_argument("--orders", type=str, help="First condition per task, e.g. 'ATA' or 'a,t,a'.")
    ap.add_argument("-m", "--metric", type=str2bool, nargs="?", const=True, default=True,
                    help="UniDepth (metric, default) vs MiDaS (relative).")
    ap.add_argument("--mock_navigate", type=str2bool, nargs="?", const=True, default=False,
                    help="Run without bracelet.")
    ap.add_argument("--source", type=str, default="1", help="Camera index / path.")
    ap.add_argument("--time_limit", type=float, default=0,
                    help="Auto-end trial as 'timeout' after N s (0 = off).")
    ap.add_argument("--shuffle_targets", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="Seed for target shuffling (default: participant).")
    ap.add_argument("--save_video", choices=["none", "raw", "annotated", "both"], default="none",
                    help="raw = clean camera feed at camera rate; annotated = what the experimenter sees.")
    ap.add_argument("--video_fps", type=float, default=15.0,
                    help="Nominal playback fps of annotated videos (exact timing is in *_timestamps.csv).")
    ap.add_argument("--no_frame_log", action="store_true", help="Disable the per-frame CSV.")
    args = ap.parse_args()

    trials = args.trials or ask_int("Trials per condition per task: ")
    block_size = args.block_size or ask_int("Trials per block: ")
    block_size = min(block_size, trials)
    orders = get_orders(args.orders)
    rng = random.Random(args.seed if args.seed is not None else args.participant)
    plan = build_plan(trials, block_size, orders, args.shuffle_targets, rng)

    print("\n=== SESSION PLAN ===")
    for task, blocks in plan:
        seq = " -> ".join(f"{b['condition'][0].upper()}{b['block']}({len(b['trials'])})" for b in blocks)
        print(f"{task:18s}: {seq}")
    print(f"Video: {args.save_video} | per-frame log: {'off' if args.no_frame_log else 'on'}")
    if input("Start with this plan? (y/n) ").strip().lower() != "y":
        sys.exit("Aborted by user.")

    out_dir = Path("results/experiment")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"participant_{args.participant}_{stamp}"
    csv_path = (out_dir / f"{prefix}_trials.csv").resolve()
    frames_path = (out_dir / f"{prefix}_frames.csv").resolve()
    video_dir = (out_dir / "videos" / prefix).resolve()

    intensities, cal_source = load_calibration(args.participant)

    with open(out_dir / f"{prefix}_config.json", "w") as f:
        json.dump(dict(vars(args), script_version=SCRIPT_VERSION, trials=trials,
                       block_size=block_size, orders=orders, targets=TASK_TARGETS,
                       intensities=intensities, calibration_source=cal_source,
                       start=stamp,
                       bracelet_sha=file_hash(root / "bracelet.py"),
                       experiment_sha=file_hash(Path(__file__).resolve()),
                       csv_path=str(csv_path), frames_path=str(frames_path),
                       video_dir=str(video_dir),
                       plan=[(t, b) for t, b in plan]), f, indent=2)

    belt_controller = None
    if not args.mock_navigate:
        ok, belt_controller = connect_belt()
        if not ok:
            sys.exit("Error connecting bracelet. Aborting.")
        print("Bracelet connection successful.")

    exp = ExperimentController(
        weights_obj="yolov5s.pt", weights_hand="hand.pt",
        weights_tracker="osnet_x0_25_market1501.pt",
        weights_depth_estimator="v2-vits14" if args.metric else "midas_v21_384",
        source=args.source, device="", imgsz=(640, 640),
        conf_thres=0.7, iou_thres=0.45, max_det=1000, agnostic_nms=False, augment=False,
        classes_obj=[1, 39, 40, 41, 42, 45, 46, 47, 58, 74], classes_hand=[0, 1],
        line_thickness=3, half=False, dnn=False,
        tracker_max_age=60, tracker_n_init=5,
        belt_controller=belt_controller, participant=args.participant,
        participant_vibration_intensities=intensities, metric=args.metric,
        time_limit=args.time_limit, csv_path=csv_path,
        frames_path=frames_path, save_frames=not args.no_frame_log,
        save_video=args.save_video, video_fps=args.video_fps, video_dir=video_dir,
    )
    try:
        exp.setup()
        exp.run_session(plan)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        exp.close()
        print(f"Trial data:  {csv_path}")
        if not args.no_frame_log:
            print(f"Frame data:  {frames_path}")
        if args.save_video != "none":
            print(f"Videos:      {video_dir}")


if __name__ == "__main__":
    main()