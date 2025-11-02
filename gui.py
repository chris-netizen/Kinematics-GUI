from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLineEdit, QLabel, QTableView,
                             QTextEdit, QGraphicsView, QGraphicsScene, QMessageBox,
                             QMenuBar, QAction, QStatusBar, QFileDialog, QComboBox)
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtGui import QFont, QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from kinematics import KinematicAnalyzer


class PandasModel(QAbstractTableModel):
    """Helper: Display DataFrame in QTableView."""

    def __init__(self, data):
        super().__init__()
        self._data = data

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def rowCount(self, index=...):
        return self._data.shape[0]

    def columnCount(self, index=...):
        return self._data.shape[1]

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            return str(self._data.index[section])
        return None


class MainWindow(QMainWindow):
    """Main GUI Window."""

    def __init__(self):
        super().__init__()
        self.analyzer = KinematicAnalyzer()
        self.canvas = None
        self.model = None
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        self.setWindowTitle("Kinematic Slope Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        theme_act = QAction('Toggle Dark Theme', self)
        theme_act.triggered.connect(self.toggle_theme)
        file_menu.addAction(theme_act)
        exit_act = QAction('Exit', self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # Tabs
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Data Tab
        data_tab = QWidget()
        data_layout = QVBoxLayout()
        load_btn = QPushButton("Load CSV")
        load_btn.clicked.connect(self.load_csv)
        data_layout.addWidget(load_btn)

        self.preview_table = QTableView()
        data_layout.addWidget(self.preview_table)

        inputs_layout = QHBoxLayout()
        self.slope_dir_edit = QLineEdit("125")
        self.slope_dip_edit = QLineEdit("45")
        self.friction_edit = QLineEdit("30")
        inputs_layout.addWidget(QLabel("Slope Dir:"))
        inputs_layout.addWidget(self.slope_dir_edit)
        inputs_layout.addWidget(QLabel("Slope Dip:"))
        inputs_layout.addWidget(self.slope_dip_edit)
        inputs_layout.addWidget(QLabel("Friction:"))
        inputs_layout.addWidget(self.friction_edit)
        data_layout.addLayout(inputs_layout)
        data_tab.setLayout(data_layout)
        tabs.addTab(data_tab, "Data")

        # Analysis Tab
        anal_tab = QWidget()
        anal_layout = QVBoxLayout()
        run_btn = QPushButton("Run Analysis")
        run_btn.clicked.connect(self.run_analysis)
        anal_layout.addWidget(run_btn)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        anal_layout.addWidget(self.results_text)

        save_btn = QPushButton("Save Results")
        save_btn.clicked.connect(self.save_results_gui)
        anal_layout.addWidget(save_btn)
        anal_tab.setLayout(anal_layout)
        tabs.addTab(anal_tab, "Analysis")

        # Plot Tab (Fixed: Use QWidget + QVBoxLayout for direct canvas embedding)
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.plot_layout = plot_layout
        self.plot_widget = plot_tab

        # Projection combo and export button
        proj_layout = QHBoxLayout()
        proj_label = QLabel("Projection:")
        self.plot_proj = QComboBox()
        self.plot_proj.addItems(["equal_area", "equal_angle"])
        self.plot_proj.currentTextChanged.connect(self.update_plot)
        proj_layout.addWidget(proj_label)
        proj_layout.addWidget(self.plot_proj)
        proj_layout.addStretch()
        plot_layout.addLayout(proj_layout)

        export_plot_btn = QPushButton("Export Plot")
        export_plot_btn.clicked.connect(self.export_plot)
        plot_layout.addWidget(export_plot_btn)

        tabs.addTab(plot_tab, "Plot")

        # Status Bar
        self.statusBar().showMessage("Ready")

    def apply_styles(self):
        """Basic styling."""
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QPushButton { background-color: #4CAF50; color: white; border: none; padding: 8px; }
            QPushButton:hover { background-color: #45a049; }
            QLineEdit { border: 1px solid #ccc; padding: 4px; }
            QTextEdit { border: 1px solid #ccc; font-family: monospace; }
        """)  # Add dark theme toggle logic here

    def toggle_theme(self):
        """Toggle dark/light (expand as needed)."""
        current = self.styleSheet()
        if 'background-color: #f0f0f0' in current:
            self.setStyleSheet(current.replace(
                '#f0f0f0', '#2b2b2b').replace('#4CAF50', '#2196F3'))
        else:
            self.apply_styles()

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load CSV", "", "CSV (*.csv)")
        if file_path:
            try:
                self.analyzer.load_data(file_path)
                self.model = PandasModel(
                    self.analyzer.df.head(10))  # Preview first 10
                self.preview_table.setModel(self.model)
                self.statusBar().showMessage(
                    f"Loaded {len(self.analyzer.df)} rows")
            except ValueError as e:
                QMessageBox.warning(self, "Error", str(e))

    def run_analysis(self):
        try:
            self.analyzer.set_parameters(
                float(self.slope_dir_edit.text() or 125),
                float(self.slope_dip_edit.text() or 45),
                float(self.friction_edit.text() or 30)
            )
            self.analyzer.analyze()
            self.results_text.setText(self.analyzer.get_summary())
            self.update_plot()
            self.statusBar().showMessage("Analysis complete")
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def update_plot(self):
        """Update stereonet plot in the Plot tab."""
        if not self.analyzer.results:
            QMessageBox.warning(self, "Error", "Run analysis first.")
            return
        proj = self.plot_proj.currentText()
        try:
            fig = self.analyzer.plot_stereonet(projection=proj)
            # Remove old canvas if exists
            if self.canvas:
                self.plot_layout.removeWidget(self.canvas)
                self.canvas.deleteLater()  # Clean up
            # Create and add new canvas
            self.canvas = FigureCanvas(fig)
            # Insert at top (before buttons)
            self.plot_layout.insertWidget(0, self.canvas)
            self.canvas.draw()  # Render the plot
            self.statusBar().showMessage(f"Stereonet updated ({proj})")
        except Exception as e:
            QMessageBox.warning(self, "Plot Error",
                                f"Failed to generate plot: {str(e)}")

    def save_results_gui(self):
        if not self.analyzer.results:
            QMessageBox.warning(self, "Error", "Run analysis first")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "results.csv", "CSV (*.csv)")
        if file_path:
            self.analyzer.save_results(file_path)
            self.statusBar().showMessage("Results saved")

    def export_plot(self):
        """Export current plot as PNG."""
        if not self.canvas or not self.canvas.figure:
            QMessageBox.warning(
                self, "Error", "No plot to export. Run analysis first.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "stereonet.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if file_path:
            try:
                self.canvas.figure.savefig(
                    file_path, dpi=300, bbox_inches='tight')
                self.statusBar().showMessage(f"Plot saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error",
                                    f"Failed to save: {str(e)}")
