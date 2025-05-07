import sys
from PyQt6.QtWidgets import QApplication
from image_scanner import ImageScannerWidget

def main():
    app = QApplication(sys.argv)
    window = ImageScannerWidget()
    window.setWindowTitle("Skaner obrazów z paletą i przezroczystością")
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 