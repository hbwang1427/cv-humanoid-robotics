import cv2

CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


class Camera:
    def __init__(self, camera_id: int = CAMERA_ID):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        # Flush first 30 frames — macOS cameras return black until warmed up
        for _ in range(30):
            self.cap.grab()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
