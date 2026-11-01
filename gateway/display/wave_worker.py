"""On-demand CPU pose worker. Frames stay in RAM; stdout contains only bounded events."""
import argparse
import json
import os
import time
import urllib.request


def read_status(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/status', timeout=.25) as response:
            raw = response.read(32769)
        return json.loads(raw) if len(raw) <= 32768 else {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    import cv2
    import numpy as np
    import mediapipe as mp
    from camera_stream_service.client import CameraStreamClient
    from .wave_detector import HandWaveDetector
    cv2.setNumThreads(2)
    os.nice(5)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path='/home/arduino/.local/share/vq-pose-lab/pose_landmarker_lite.task',
            delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=2,
        min_pose_detection_confidence=.65, min_pose_presence_confidence=.65,
        min_tracking_confidence=.65, output_segmentation_masks=False)
    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path='/home/arduino/.local/share/vq-pose-lab/hand_landmarker-starbuddy-20261019.task',
            delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=.7, min_hand_presence_confidence=.7,
        min_tracking_confidence=.7)
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model, \
            mp.tasks.vision.HandLandmarker.create_from_options(hand_options) as hand_model:
        if args.self_test:
            blank = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((240, 320, 3), np.uint8))
            started = time.monotonic()
            result = model.detect_for_video(blank, 1)
            hand_result = hand_model.detect_for_video(blank, 1)
            print(json.dumps({'self_test': True, 'poses': len(result.pose_landmarks),
                              'hands': len(hand_result.hand_landmarks),
                              'elapsed_ms': round((time.monotonic()-started)*1000, 2)}), flush=True)
            return
        camera = CameraStreamClient(consumer='wave-fusion', source='camera.front', rotation=0, timeout=1)
        detector, sequence, frames, last_report = HandWaveDetector(), 0, 0, 0
        while True:
            started = time.monotonic()
            value = camera.wait_frame(sequence, timeout=1)
            ready, event, latency, present, captured = False, None, None, False, 0
            if value:
                sequence, jpeg = value
                pixels = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if pixels is not None:
                    pixels = cv2.rotate(pixels, cv2.ROTATE_180)
                    height, width = pixels.shape[:2]
                    pixels = cv2.resize(pixels, (min(width, 480), round(height*min(width, 480)/width)))
                    frame = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                    captured = time.time()
                    result = model.detect_for_video(frame, int(time.monotonic()*1000))
                    hand_result = hand_model.detect_for_video(frame, int(time.monotonic()*1000))
                    present = len(result.pose_landmarks) == 1 and all(
                        result.pose_landmarks[0][i].visibility >= .8 for i in (0,11,12))
                    face, identity = read_status(8794), read_status(8795)
                    event = detector.feed_hands(result.pose_landmarks, hand_result.hand_landmarks,
                        face, identity, captured, args.generation, association_at=time.time())
                    del pixels, frame, jpeg, result, hand_result, face, identity
                    ready, frames = True, frames+1
                    latency = round((time.monotonic()-started)*1000, 1)
            if not ready:
                detector.reset()
            if event or time.monotonic()-last_report >= .5:
                print(json.dumps({'type': 'status', 'ready': ready, 'frames': frames,
                                  'latency_ms': latency, 'person_present': present,
                                  'observed_at': captured, 'body_epoch': detector.epoch,
                                  'detector': detector.diagnostics}), flush=True)
                last_report = time.monotonic()
            if event:
                print(json.dumps({'type': 'event', 'event': event}), flush=True)
            # Bound compute rate; no inference backlog, recordings or media storage.
            time.sleep(max(0, 1/6-(time.monotonic()-started)))


if __name__ == '__main__':
    main()
