import cv2
import numpy as np
from ultralytics import YOLO


class ParkingDetection:

    def __init__(self, video_path: str, mask_path: str, model_path: str):
        """
        :param video_path: Path of the video to be processed.
        :param mask_path: Path of the video mask image.
        :param model_path: Path of the model
        """
        self.video_path = video_path
        self.mask_path = mask_path
        self.model = YOLO(model_path)

        self.ret = True

        self.cap = cv2.VideoCapture(self.video_path)
        self.mask = cv2.imread(self.mask_path, 0)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter("result.mp4", self.fourcc, self.fps, (self.width, self.height))

    def prediction(self, total_area: dict) -> dict:
        """
        :param total_area: View of all parking areas.
        :return: Occupancy information of all parking areas.
        """
        data = []
        y_area = {}

        for area in total_area.values():
            x, y, _ = area.shape
            if x < y:
                area = cv2.rotate(area, cv2.ROTATE_90_CLOCKWISE)
            data.append(area)

        results = self.model.predict(data, save=False, verbose=False)

        for num, res in enumerate(results):
            output = res.probs.data.argmax().cpu().numpy()
            y_area[str(num)] = output

        return y_area

    @staticmethod
    def get_spots(connected_components) -> list:
        """
        :param connected_components: Connected components.
        :return: Coordinates of parking areas.
        """
        (totalLabels, label_ids, values, centroid) = connected_components

        spots = []
        coef = 1
        for i in range(1, totalLabels):
            x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
            y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
            w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
            h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

            spots.append([x1, y1, w, h])

        return spots

    @staticmethod
    def draw_statu(frame: np.ndarray, status: dict, spots: list) -> np.ndarray:
        """
        :param frame: Image of source video.
        :param status: Occupancy information of all parking areas.
        :param spots: Coordinates of parking areas.
        :return: Processed image.
        """
        count = 0
        for spot, statu in zip(spots, status.values()):
            x1, y1, w, h = spot
            x1, y1 = x1+1, y1+1
            w, h = w-1, h-1

            if statu:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                frame[y1:y1+h, x1:x1+w, 2] += 75
                count += 1
            else:
                frame[y1:y1+h, x1:x1+w, 0] += 75
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)

        occ_text = str(count) + " / " + str(len(spots))
        cv2.putText(frame, occ_text, (725, 700), 0, 2, (0, 0, 0), 5)

        return frame

    def detection(self) -> None:
        """
        :return: None
        """
        i = 0
        status = []

        connected_components = cv2.connectedComponentsWithStats(self.mask, 4, cv2.CV_32S)
        spots = self.get_spots(connected_components)

        while self.ret:
            self.ret, frame = self.cap.read()
            i += 1
            total_area = {}
            if self.ret:
                if (i % 25 == 0) or (i == 1):
                    for num, spot in enumerate(spots):
                        x1, y1, w, h = spot
                        total_area[str(num)] = frame[y1:y1 + h, x1:x1 + w, :]

                    status = self.prediction(total_area)
                    frame = self.draw_statu(frame, status, spots)

                else:
                    frame = self.draw_statu(frame, status, spots)

                self.out.write(frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
