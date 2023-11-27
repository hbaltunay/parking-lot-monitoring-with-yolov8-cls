from core import ParkingDetection

if __name__ == "__main__":

    video_path = "./video.mp4"
    mask_path = "./mask.png"
    model_path = "./weights/last.pt"

    PDetection = ParkingDetection(
        video_path=video_path,
        mask_path=mask_path,
        model_path=model_path,
    )

    PDetection.detection()
