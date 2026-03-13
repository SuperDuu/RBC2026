import cv2
import numpy as np
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class RobotPreprocessing:
    def __init__(self, device_path="/dev/video0", clip_limit=1.5, tile_size=(8, 8)):
        self.device_path = device_path
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        self.denoise_kernel = np.ones((3, 3), np.uint8)
        self.lock_hardware_controls()

    def lock_hardware_controls(self):
        try:
            subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', 'white_balance_automatic=0'], check=True)
            subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', 'white_balance_temperature=4600'], check=True)
            subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', 'auto_exposure=1'], check=True)
            subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', 'exposure_time_absolute=600'], check=True)
            subprocess.run(['v4l2-ctl', '-d', self.device_path, '-c', 'gain=3'], check=True)
            logging.info("--- HARDWARE LOCKED SUCCESSFULLY: MANUAL MODE ENABLED ---")
        except Exception as e:
            logging.error(f"Lỗi khóa phần cứng: {e}.")

    def enhance_contrast_adaptive(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        enhanced_lab = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def quick_denoise(self, frame):
        return cv2.morphologyEx(frame, cv2.MORPH_OPEN, self.denoise_kernel)

    def run_pipeline(self, frame):
        if frame is None:
            return None
            
        frame = self.enhance_contrast_adaptive(frame)
        frame = self.quick_denoise(frame)
        
        return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    preprocessor = RobotPreprocessing()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        processed_frame = preprocessor.run_pipeline(frame)
        
        cv2.imshow("Original", frame)
        cv2.imshow("Preprocessed", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()