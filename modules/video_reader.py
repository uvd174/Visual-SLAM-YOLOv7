import cv2


class VideoReader(object):
    def __init__(self, file_name: str, mode: str = 'BGR'):
        assert mode in ('BGR', 'GRAY'), 'mode should be BGR or GRAY'

        self.file_name = file_name
        self.cap = cv2.VideoCapture(self.file_name)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.mode = mode

    def __len__(self):
        return self.length

    def __iter__(self):
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()

        if not was_read:
            raise StopIteration

        if self.mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img
