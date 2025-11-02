# gui/tabs/plot_tab.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QCheckBox, QComboBox, QMessageBox, QFileDialog
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from core.kinematics import KinematicAnalyzer


class PlotTab(QWidget):
    def __init__(self, data_tab, analysis_tab):
        super().__init__()
        self.data_tab = data_tab
        self.analysis_tab = analysis_tab
        self.canvas = None
        self.toolbar = None
        self.analyzer = analysis_tab.analyzer
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        opts = QHBoxLayout()
        self.contours_cb = QCheckBox("Contours")
        self.contours_cb.setChecked(True)
        self.clusters_cb = QCheckBox("Show Clusters")
        self.clusters_cb.setChecked(True)
        self.friction_cb = QCheckBox("Friction Cone")
        self.friction_cb.setChecked(True)
        opts.addWidget(self.contours_cb)
        opts.addWidget(self.clusters_cb)
        opts.addWidget(self.friction_cb)
        opts.addStretch()
        layout.addLayout(opts)

        proj_layout = QHBoxLayout()
        self.proj_combo = QComboBox()
        self.proj_combo.addItems(["equal_area", "equal_angle"])
        proj_layout.addWidget(self.proj_combo)
        self.plot_btn = QPushButton("Update Plot")
        self.plot_btn.clicked.connect(self.update_plot)
        proj_layout.addWidget(self.plot_btn)
        layout.addLayout(proj_layout)

        self.export_btn = QPushButton("Export Plot")
        self.export_btn.clicked.connect(self.export_plot)
        layout.addWidget(self.export_btn)

    def update_plot(self):
        if not self.analyzer.results:
            QMessageBox.warning(self, "No Results", "Run analysis first")
            return
        proj = self.proj_combo.currentText()
        fig = self.analyzer.plot_stereonet(projection=proj,
                                           show_contours=self.contours_cb.isChecked(),
                                           show_clusters=self.clusters_cb.isChecked(),
                                           show_friction=self.friction_cb.isChecked())
        # attach to canvas
        if self.canvas:
            self.canvas.setParent(None)
        self.canvas = FigureCanvas(fig)
        # add toolbar to the widget layout (simple approach)
        if self.toolbar:
            self.toolbar.setParent(None)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)
        self.canvas.draw()

    def export_plot(self):
        if not self.canvas:
            QMessageBox.warning(self, "No Plot", "Generate a plot first")
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "stereonet.png", "PNG (*.png);;PDF (*.pdf)")
        if not p:
            return
        try:
            self.canvas.figure.savefig(p, dpi=300, bbox_inches='tight')
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))
