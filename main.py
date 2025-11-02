# main.py
import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from pathlib import Path

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # load external qss
    qss_path = Path(__file__).parent / "resources" / "style.qss"
    if qss_path.exists():
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
