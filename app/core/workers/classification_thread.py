from PyQt6.QtCore import QThread, pyqtSignal


class ClassificationThread(QThread):
    """Wątek do wykonywania klasyfikacji w tle."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, classifier, image_path):
        super().__init__()
        self.classifier = classifier
        self.image_path = image_path

    def run(self):
        try:
            result = self.classifier.predict(self.image_path)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
