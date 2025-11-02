import sys
from gui import MainWindow
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLineEdit, QLabel, QTableView,
                             QTextEdit, QGraphicsView, QGraphicsScene, QMessageBox,
                             QMenuBar, QAction, QStatusBar, QFileDialog, QComboBox)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
