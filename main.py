from pose_detection import PoseDetector
from audio_processing import MicController
import threading

def pose_detection_thread():
    detector = PoseDetector()
    detector.start()

def audio_processing_thread():
    controller = MicController()
    controller.process_and_respond_standalone()

def main():
    pose_thread = threading.Thread(target=pose_detection_thread)
    audio_thread = threading.Thread(target=audio_processing_thread)
    
    pose_thread.start()
    audio_thread.start()
    
    pose_thread.join()
    audio_thread.join()

if __name__ == "__main__":
    main()
