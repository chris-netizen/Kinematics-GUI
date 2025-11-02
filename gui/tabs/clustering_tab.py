# gui/tabs/clustering_tab.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QMessageBox
from PySide6.QtCore import Qt, Slot
from core.worker import Worker
from core.kinematics import KinematicAnalyzer


class ClusteringTab(QWidget):
    def __init__(self, data_tab):
        super().__init__()
        self.data_tab = data_tab
        self.analyzer = KinematicAnalyzer()
        self._init_ui()
        self.worker = None

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.run_btn = QPushButton("Run Clustering (background)")
        self.run_btn.clicked.connect(self.run_clustering)
        layout.addWidget(self.run_btn)
        self.stats = QTextEdit()
        self.stats.setReadOnly(True)
        layout.addWidget(self.stats)

    def run_clustering(self):
        if self.data_tab.df is None:
            QMessageBox.warning(self, "No Data", "Load data first")
            return
        self.run_btn.setEnabled(False)
        # prepare analyzer
        self.analyzer.df = self.data_tab.df.copy()
        n_clusters = int(self.data_tab.n_clusters.value())
        self.analyzer.n_clusters = n_clusters

        def job():
            self.analyzer.perform_clustering()
            return self.analyzer.get_cluster_stats()

        def on_done(result):
            self.stats.setPlainText(result)
            self.run_btn.setEnabled(True)

        def on_error(exc):
            QMessageBox.warning(self, "Clustering Error", str(exc))
            self.run_btn.setEnabled(True)

        self.worker = Worker(job, on_done, on_error)
        self.worker.start()
