# gui/main_window.py
from PySide6.QtWidgets import QMainWindow, QTabWidget, QMessageBox
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from gui.tabs.data_tab import DataTab
from gui.tabs.clustering_tab import ClusteringTab
from gui.tabs.analysis_tab import AnalysisTab
from gui.tabs.plot_tab import PlotTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinematic Slope Analyzer (Refactor)")
        self.setGeometry(100, 100, 1400, 900)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create Tabs
        self.data_tab = DataTab()
        self.clustering_tab = ClusteringTab(self.data_tab)
        self.analysis_tab = AnalysisTab(self.data_tab)
        self.plot_tab = PlotTab(self.data_tab, self.analysis_tab)

        # Add tabs
        self.tabs.addTab(self.data_tab, "Data")
        self.tabs.addTab(self.clustering_tab, "Clustering")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.plot_tab, "Plot")

        # Menu (basic)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        theme_act = QAction("Toggle Dark Theme", self)
        theme_act.triggered.connect(self.toggle_theme)
        file_menu.addAction(theme_act)

    def toggle_theme(self):
        # small theme toggle demo (swap stylesheet if loaded)
        current = self.styleSheet()
        if "background-color: #f0f0f0" in current:
            self.setStyleSheet(current.replace("#f0f0f0", "#2b2b2b"))
        else:
            # reload qss from resources
            pass
