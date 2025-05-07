import sys

from PyQt6.QtWidgets import QApplication

from app.utils.resolution_scanner import ResolutionScannerWidget


def main():
    app = QApplication(sys.argv)
    window = ResolutionScannerWidget()
    window.setWindowTitle("Skaner rozdzielczości obrazów")
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
