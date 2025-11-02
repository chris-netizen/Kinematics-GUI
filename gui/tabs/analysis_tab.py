# gui/tabs/analysis_tab.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QMessageBox
from core.worker import Worker
from core.kinematics import KinematicAnalyzer


class AnalysisTab(QWidget):
    def __init__(self, data_tab):
        super().__init__()
        self.data_tab = data_tab
        self.analyzer = KinematicAnalyzer()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.run_btn = QPushButton("Run Analysis (background)")
        self.run_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.run_btn)
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        layout.addWidget(self.results)

    def run_analysis(self):
        if self.data_tab.df is None:
            QMessageBox.warning(self, "No Data", "Load data first")
            return
        self.run_btn.setEnabled(False)
        # configure analyzer from UI
        self.analyzer.df = self.data_tab.df.copy()
        try:
            slope_dir = float(self.data_tab.slope_dir.text())
            slope_dip = float(self.data_tab.slope_dip.text())
            friction = float(self.data_tab.friction.text())
            n_clusters = int(self.data_tab.n_clusters.value())
            self.analyzer.set_parameters(
                slope_dir, slope_dip, friction, n_clusters=n_clusters)
        except Exception as e:
            QMessageBox.warning(self, "Parameter Error", str(e))
            self.run_btn.setEnabled(True)
            return

        def job():
            return self.analyzer.analyze()

        def on_done(result):
            summary = self.analyzer.get_summary()
            self.results.setPlainText(summary)
            self.run_btn.setEnabled(True)

        def on_error(exc):
            QMessageBox.warning(self, "Analysis Error", str(exc))
            self.run_btn.setEnabled(True)

        worker = Worker(job, on_done, on_error)
        worker.start()
