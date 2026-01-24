#!/usr/bin/env python

"""
Opens a camera by index and displays a live stream window.

Example:

```shell
python -m lerobot.scripts.lerobot_open_cameras 0
```

Press 'q' or ESC to close the window.
"""

import argparse
import platform

import cv2


def get_backend() -> int:
    if platform.system() == "Windows":
        return int(cv2.CAP_DSHOW)
    else:
        return int(cv2.CAP_ANY)


def main():
    parser = argparse.ArgumentParser(description="Open a camera by index and display a live stream.")
    parser.add_argument("index", type=int, help="Camera index to open (e.g., 0, 1, 2)")
    parser.add_argument("--width", type=int, default=None, help="Capture width")
    parser.add_argument("--height", type=int, default=None, help="Capture height")
    parser.add_argument("--fps", type=int, default=None, help="Capture FPS")
    args = parser.parse_args()

    backend = get_backend()
    cap = cv2.VideoCapture(args.index, backend)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {args.index}")
        return

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera {args.index} opened: {width}x{height} @ {fps:.1f} FPS")
    print("Press 'q' or ESC to quit.")

    window_name = f"Camera {args.index}"
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
