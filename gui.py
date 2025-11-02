from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLineEdit, QLabel, QTableView,
                             QTextEdit, QGraphicsView, QGraphicsScene, QMessageBox,
                             QMenuBar, QAction, QStatusBar, QFileDialog, QComboBox,
                             QCheckBox, QSpinBox, QHeaderView, QDialog, QFormLayout,
                             QInputDialog)
from PyQt5.QtCore import Qt, QAbstractTableModel, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QCursor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge  # For rose

from kinematics import KinematicAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PandasModel(QAbstractTableModel):
    """Display/edit DataFrame in QTableView."""
    data_changed = pyqtSignal()  # New: Signal for edits

    def __init__(self, data):
        super().__init__()
        self._data = data

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole or role == Qt.EditRole:
            val = self._data.iloc[index.row(), index.column()]
            return str(val) if pd.notna(val) else ''
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            try:
                self._data.iloc[index.row(), index.column()] = pd.to_numeric(value) if self._data.dtypes[index.column(
                )] == 'float64' or self._data.dtypes[index.column()] == 'int64' else value
                self.data_changed.emit()
                self.layoutChanged.emit()
                return True
            except:
                return False
        return False

    def flags(self, index):
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

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


class ColumnMapperDialog(QDialog):
    """Dialog to select dip direction and dip angle columns."""

    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Columns for Analysis")
        layout = QFormLayout()
        self.dip_dir_combo = QComboBox()
        self.dip_dir_combo.addItems(columns)
        self.dip_combo = QComboBox()
        self.dip_combo.addItems(columns)
        layout.addRow("Dip Direction Column:", self.dip_dir_combo)
        layout.addRow("Dip Angle Column:", self.dip_combo)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)
        self.setLayout(layout)

    def get_selections(self):
        return self.dip_dir_combo.currentText(), self.dip_combo.currentText()


class MainWindow(QMainWindow):
    """Enhanced GUI: Clustering tab, interactive plot, manual edit, flexible columns."""

    def __init__(self):
        super().__init__()
        self.analyzer = KinematicAnalyzer()
        self.canvas = None
        self.model = None
        self.current_plot_type = 'Stereonet'
        self.x_col = None
        self.y_col = None
        self.bar_col = None
        self.init_ui()
        self.apply_styles()
        # Set initial plot type visibility
        self.on_plot_type_changed("Stereonet")

    def init_ui(self):
        self.setWindowTitle("Enhanced Kinematic Slope Analyzer (DIPS-like)")
        self.setGeometry(100, 100, 1400, 900)

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

        # Data Tab: Load + Manual Edit + Column Management
        data_tab = QWidget()
        data_layout = QVBoxLayout()
        load_btn = QPushButton("Load Data File")
        load_btn.clicked.connect(self.load_file)
        data_layout.addWidget(load_btn)

        # Column management
        col_layout = QHBoxLayout()
        add_col_btn = QPushButton("Add Column")
        add_col_btn.clicked.connect(self.add_column)
        col_layout.addWidget(add_col_btn)
        map_col_btn = QPushButton("Map Dip Columns")
        map_col_btn.clicked.connect(self.map_columns)
        col_layout.addWidget(map_col_btn)
        data_layout.addLayout(col_layout)

        # Manual entry
        manual_layout = QHBoxLayout()
        add_row_btn = QPushButton("Add Row")
        add_row_btn.clicked.connect(self.add_row)
        manual_layout.addWidget(add_row_btn)
        clear_btn = QPushButton("Clear Data")
        clear_btn.clicked.connect(self.clear_data)
        manual_layout.addWidget(clear_btn)
        data_layout.addLayout(manual_layout)

        self.preview_table = QTableView()
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_table.setAlternatingRowColors(True)
        data_layout.addWidget(self.preview_table)

        # Inputs (enhanced with n_clusters)
        inputs_layout = QHBoxLayout()
        self.slope_dir_edit = QLineEdit("125")
        self.slope_dip_edit = QLineEdit("45")
        self.friction_edit = QLineEdit("30")
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(1, 10)
        self.n_clusters_spin.setValue(3)
        inputs_layout.addWidget(QLabel("Slope Dir:"))
        inputs_layout.addWidget(self.slope_dir_edit)
        inputs_layout.addWidget(QLabel("Slope Dip:"))
        inputs_layout.addWidget(self.slope_dip_edit)
        inputs_layout.addWidget(QLabel("Friction:"))
        inputs_layout.addWidget(self.friction_edit)
        inputs_layout.addWidget(QLabel("N Clusters:"))
        inputs_layout.addWidget(self.n_clusters_spin)
        data_layout.addLayout(inputs_layout)
        data_tab.setLayout(data_layout)
        tabs.addTab(data_tab, "Data")

        # New: Clustering Tab
        cluster_tab = QWidget()
        cluster_layout = QVBoxLayout()
        cluster_btn = QPushButton("Run Clustering")
        cluster_btn.clicked.connect(self.run_clustering)
        cluster_layout.addWidget(cluster_btn)

        self.cluster_stats_text = QTextEdit()
        self.cluster_stats_text.setReadOnly(True)
        cluster_layout.addWidget(self.cluster_stats_text)
        cluster_tab.setLayout(cluster_layout)
        tabs.addTab(cluster_tab, "Clustering")

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

        # Plot Tab: Enhanced with general plotting options
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.plot_layout = plot_layout  # For adding canvas

        # Plot type selection
        plot_type_layout = QHBoxLayout()
        plot_type_label = QLabel("Plot Type:")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(
            ["Stereonet", "Scatter Plot", "Bar Chart"])
        self.plot_type_combo.currentTextChanged.connect(
            self.on_plot_type_changed)
        plot_type_layout.addWidget(plot_type_label)
        plot_type_layout.addWidget(self.plot_type_combo)
        plot_type_layout.addStretch()
        self.plot_layout.addLayout(plot_type_layout)

        # Column selections
        self.col_sel_layout = QHBoxLayout()
        self.col_x_label = QLabel("X Col:")
        self.col_sel_layout.addWidget(self.col_x_label)
        self.x_col_combo = QComboBox()
        self.col_sel_layout.addWidget(self.x_col_combo)
        self.col_y_label = QLabel("Y Col:")
        self.col_sel_layout.addWidget(self.col_y_label)
        self.y_col_combo = QComboBox()
        self.col_sel_layout.addWidget(self.y_col_combo)
        self.col_bar_label = QLabel("Bar Col:")
        self.col_sel_layout.addWidget(self.col_bar_label)
        self.bar_col_combo = QComboBox()
        self.col_sel_layout.addWidget(self.bar_col_combo)
        self.plot_layout.addLayout(self.col_sel_layout)

        # Stereonet options
        self.stereo_options_layout = QHBoxLayout()
        self.show_contours_cb = QCheckBox("Contours (Fuchsia)")
        self.show_contours_cb.setChecked(True)
        self.show_contours_cb.stateChanged.connect(self.update_plot)
        self.stereo_options_layout.addWidget(self.show_contours_cb)
        self.show_clusters_cb = QCheckBox("Show Clusters")
        self.show_clusters_cb.setChecked(True)
        self.show_clusters_cb.stateChanged.connect(self.update_plot)
        self.stereo_options_layout.addWidget(self.show_clusters_cb)
        self.show_friction_cb = QCheckBox("Friction Cone")
        self.show_friction_cb.setChecked(True)
        self.show_friction_cb.stateChanged.connect(self.update_plot)
        self.stereo_options_layout.addWidget(self.show_friction_cb)
        self.stereo_options_layout.addStretch()
        self.plot_layout.addLayout(self.stereo_options_layout)

        # Projection for stereonet
        self.proj_layout = QHBoxLayout()
        self.proj_label = QLabel("Projection:")
        self.proj_layout.addWidget(self.proj_label)
        self.plot_proj = QComboBox()
        self.plot_proj.addItems(["equal_area", "equal_angle"])
        self.plot_proj.currentTextChanged.connect(self.update_plot)
        self.proj_layout.addWidget(self.plot_proj)

        rose_btn = QPushButton("Generate Rose Diagram")
        rose_btn.clicked.connect(self.plot_rose)
        self.proj_layout.addWidget(rose_btn)
        self.plot_layout.addLayout(self.proj_layout)

        # Update button
        update_plot_btn = QPushButton("Update Plot")
        update_plot_btn.clicked.connect(self.update_plot)
        self.plot_layout.addWidget(update_plot_btn)

        export_plot_btn = QPushButton("Export Plot")
        export_plot_btn.clicked.connect(self.export_plot)
        self.plot_layout.addWidget(export_plot_btn)

        # Initially hide column selection widgets
        self.col_x_label.hide()
        self.x_col_combo.hide()
        self.col_y_label.hide()
        self.y_col_combo.hide()
        self.col_bar_label.hide()
        self.bar_col_combo.hide()

        tabs.addTab(plot_tab, "Plot")

        # Status Bar
        self.statusBar().showMessage("Ready")
        self.current_hover = ""  # For interactive

    def on_plot_type_changed(self, plot_type):
        self.current_plot_type = plot_type
        if self.analyzer.df is not None:
            columns = self.analyzer.df.columns.tolist()
            self.x_col_combo.clear()
            self.y_col_combo.clear()
            self.bar_col_combo.clear()
            self.x_col_combo.addItems(columns)
            self.y_col_combo.addItems(columns)
            self.bar_col_combo.addItems(columns)

        # Show/hide based on plot type
        if plot_type == "Stereonet":
            self.col_x_label.hide()
            self.x_col_combo.hide()
            self.col_y_label.hide()
            self.y_col_combo.hide()
            self.col_bar_label.hide()
            self.bar_col_combo.hide()
            self.show_stereo_options()
            self.show_proj()
        elif plot_type == "Scatter Plot":
            self.col_x_label.show()
            self.x_col_combo.show()
            self.col_y_label.show()
            self.y_col_combo.show()
            self.col_bar_label.hide()
            self.bar_col_combo.hide()
            self.hide_stereo_options()
            self.hide_proj()
        else:  # Bar Chart
            self.col_x_label.hide()
            self.x_col_combo.hide()
            self.col_y_label.hide()
            self.y_col_combo.hide()
            self.col_bar_label.show()
            self.bar_col_combo.show()
            self.hide_stereo_options()
            self.hide_proj()
        self.update_plot()

    def show_stereo_options(self):
        for i in range(self.stereo_options_layout.count()):
            item = self.stereo_options_layout.itemAt(i)
            if item and item.widget():
                item.widget().show()

    def hide_stereo_options(self):
        for i in range(self.stereo_options_layout.count()):
            item = self.stereo_options_layout.itemAt(i)
            if item and item.widget():
                item.widget().hide()

    def show_proj(self):
        self.proj_label.show()
        self.plot_proj.show()

    def hide_proj(self):
        self.proj_label.hide()
        self.plot_proj.hide()

    def apply_styles(self):
        """Light theme."""
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QPushButton { background-color: #4CAF50; color: white; border: none; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #45a049; }
            QLineEdit, QSpinBox { border: 1px solid #ccc; padding: 4px; border-radius: 2px; }
            QTextEdit { border: 1px solid #ccc; font-family: monospace; background: white; }
            QTableView { gridline-color: #ccc; background-color: white; }
            QCheckBox { color: #333; }
        """)

    def toggle_theme(self):
        """Toggle dark/light."""
        current = self.styleSheet()
        if 'background-color: #f0f0f0' in current:
            dark_style = current.replace('#f0f0f0', '#2b2b2b').replace('#4CAF50', '#2196F3').replace(
                'background: white', 'background: #3c3c3c').replace('#ccc', '#555')
            self.setStyleSheet(dark_style)
        else:
            self.apply_styles()

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Data", "", "Data Files (*.csv *.xlsx *.xls)")
        if file_path:
            try:
                self.analyzer.load_data(file_path)
                self.model = PandasModel(self.analyzer.df)
                self.preview_table.setModel(self.model)
                self.model.data_changed.connect(self.on_data_changed)
                self.statusBar().showMessage(
                    f"Loaded {len(self.analyzer.df)} rows with {len(self.analyzer.df.columns)} columns")
                # Auto-map if default columns exist
                if 'Dip_Direction' in self.analyzer.df.columns and 'Dip_Angle' in self.analyzer.df.columns:
                    self.analyzer.set_dip_columns('Dip_Direction', 'Dip_Angle')
                else:
                    self.map_columns()
            except ValueError as e:
                QMessageBox.warning(self, "Error", str(e))

    def map_columns(self):
        if self.analyzer.df is None:
            return
        dialog = ColumnMapperDialog(self.analyzer.df.columns.tolist(), self)
        if dialog.exec_() == QDialog.Accepted:
            dip_dir, dip = dialog.get_selections()
            try:
                self.analyzer.set_dip_columns(dip_dir, dip)
                self.statusBar().showMessage(
                    f"Columns mapped: {dip_dir}/{dip}")
                # Update combos
                columns = self.analyzer.df.columns.tolist()
                self.x_col_combo.clear()
                self.y_col_combo.clear()
                self.bar_col_combo.clear()
                self.x_col_combo.addItems(columns)
                self.y_col_combo.addItems(columns)
                self.bar_col_combo.addItems(columns)
            except ValueError as e:
                QMessageBox.warning(self, "Mapping Error", str(e))

    def add_column(self):
        if self.analyzer.df is None:
            return
        col_name, ok = QInputDialog.getText(
            self, "Add Column", "Enter column name:")
        if ok and col_name:
            if col_name in self.analyzer.df.columns:
                QMessageBox.warning(self, "Error", "Column already exists.")
                return
            self.analyzer.df[col_name] = np.nan
            self.model = PandasModel(self.analyzer.df)
            self.preview_table.setModel(self.model)
            self.model.layoutChanged.emit()
            # Update combos
            columns = self.analyzer.df.columns.tolist()
            self.x_col_combo.addItem(col_name)
            self.y_col_combo.addItem(col_name)
            self.bar_col_combo.addItem(col_name)
            self.statusBar().showMessage(f"Added column: {col_name}")

    def add_row(self):
        """Add empty row for manual entry."""
        if self.model is None:
            default_cols = ['Dip_Direction', 'Dip_Angle']
            self.analyzer.df = pd.DataFrame(columns=default_cols)
            self.model = PandasModel(self.analyzer.df)
            self.preview_table.setModel(self.model)
            self.model.data_changed.connect(self.on_data_changed)
        new_row = pd.DataFrame(
            {col: np.nan for col in self.analyzer.df.columns}, index=[0])
        self.analyzer.df = pd.concat(
            [self.analyzer.df, new_row], ignore_index=True)
        self.model = PandasModel(self.analyzer.df)
        self.preview_table.setModel(self.model)
        self.model.layoutChanged.emit()

    def clear_data(self):
        self.analyzer.df = None
        self.model = None
        self.preview_table.setModel(None)
        self.statusBar().showMessage("Data cleared")

    def on_data_changed(self):
        """Update analyzer on edit."""
        if self.model is not None:
            self.analyzer.df = self.model._data.copy()
            try:
                if hasattr(self.analyzer, 'dip_dir_col') and self.analyzer.dip_dir_col in self.analyzer.df.columns:
                    self.analyzer._validate_data()
            except ValueError as e:
                QMessageBox.warning(self, "Validation Error", str(e))

    def run_clustering(self):
        """Run clustering and show stats."""
        try:
            self.analyzer.n_clusters = self.n_clusters_spin.value()
            self.analyzer.perform_clustering()
            self.cluster_stats_text.setText(self.analyzer.get_cluster_stats())
            self.statusBar().showMessage(
                f"Clustered into {self.analyzer.n_clusters} sets")
        except Exception as e:
            QMessageBox.warning(self, "Clustering Error", str(e))

    def run_analysis(self):
        try:
            self.analyzer.set_parameters(
                float(self.slope_dir_edit.text() or 125),
                float(self.slope_dip_edit.text() or 45),
                float(self.friction_edit.text() or 30),
                self.n_clusters_spin.value()
            )
            self.analyzer.analyze()
            self.results_text.setText(self.analyzer.get_summary())
            self.update_plot()
            self.statusBar().showMessage("Analysis complete")
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def update_plot(self):
        """Update plot based on type."""
        if self.analyzer.df is None:
            QMessageBox.warning(self, "Error", "Load data first.")
            return
        try:
            if self.current_plot_type == "Stereonet":
                if not self.analyzer.results:
                    QMessageBox.warning(self, "Error", "Run analysis first.")
                    return
                proj = self.plot_proj.currentText()
                fig = self.analyzer.plot_stereonet(
                    projection=proj,
                    show_contours=self.show_contours_cb.isChecked(),
                    show_clusters=self.show_clusters_cb.isChecked(),
                    show_friction=self.show_friction_cb.isChecked()
                )
            elif self.current_plot_type == "Scatter Plot":
                x = self.x_col_combo.currentText()
                y = self.y_col_combo.currentText()
                if not x or not y:
                    QMessageBox.warning(
                        self, "Error", "Select X and Y columns.")
                    return
                fig = self.analyzer.plot_scatter(x, y)
            else:  # Bar Chart
                col = self.bar_col_combo.currentText()
                if not col:
                    QMessageBox.warning(
                        self, "Error", "Select column for bar chart.")
                    return
                fig = self.analyzer.plot_bar(col)

            if self.canvas:
                self.plot_layout.removeWidget(self.canvas)
                self.canvas.deleteLater()
            # Remove old toolbar if exists
            for i in reversed(range(self.plot_layout.count())):
                child = self.plot_layout.itemAt(i).widget()
                if child and isinstance(child, NavigationToolbar):
                    self.plot_layout.removeWidget(child)
                    child.deleteLater()
            self.canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(self.canvas, self)
            self.plot_layout.addWidget(toolbar)
            self.plot_layout.addWidget(self.canvas)
            if self.current_plot_type == "Stereonet":
                fig.canvas.mpl_connect(
                    'motion_notify_event', self.on_mouse_move)
            self.canvas.draw()
            self.statusBar().showMessage(
                f"Plot updated: {self.current_plot_type}")
        except Exception as e:
            QMessageBox.warning(self, "Plot Error",
                                f"Failed to generate plot: {str(e)}")

    def on_mouse_move(self, event):
        """Interactive hover: Update status on mouse move (stereonet only)."""
        if self.current_plot_type != "Stereonet" or not event.inaxes or not hasattr(self.canvas.figure, 'hover_data'):
            return
        x, y = event.xdata, event.ydata
        tol = getattr(self.canvas.figure, 'hover_tolerance', 0.05)
        hover_data = self.canvas.figure.hover_data
        for (hx, hy), infos in hover_data.items():
            if abs(hx - x) < tol and abs(hy - y) < tol:
                info = infos[0]
                if info['type'] == 'pole':
                    self.statusBar().showMessage(
                        f"Pole {info['index']}: {info['dip_dir']:.1f}°/{info['dip']:.1f}° | FS: {info.get('fs', 'N/A')}")
                elif info['type'] == 'wedge':
                    self.statusBar().showMessage(
                        f"Wedge ({info['planes']}): {info['trend']:.1f}°/{info['plunge']:.1f}° | FS: {info['fs']:.2f}")
                else:
                    self.statusBar().showMessage(
                        f"Slope: {info.get('dir', 'N/A'):.1f}°/{info.get('dip', 'N/A'):.1f}°")
                return
        self.statusBar().showMessage("Hover over elements for details")

    def plot_rose(self):
        """Simple rose diagram for strikes using dip dir column."""
        if self.analyzer.df is None:
            QMessageBox.warning(self, "Error", "Load data first.")
            return
        if not hasattr(self.analyzer, 'dip_dir_col') or self.analyzer.dip_dir_col not in self.analyzer.df.columns:
            QMessageBox.warning(self, "Error", "Map dip columns first.")
            return
        strikes = self.analyzer.df[self.analyzer.dip_dir_col] - 90
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.hist(np.radians(strikes), bins=36, alpha=0.7, color='green')
        ax.set_title('Strike Rose Diagram')
        canvas = FigureCanvas(fig)
        canvas.show()
        plt.close(fig)

    def save_results_gui(self):
        if not self.analyzer.results:
            QMessageBox.warning(self, "Error", "Run analysis first")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "results.csv", "CSV (*.csv)")
        if file_path:
            try:
                self.analyzer.save_results(file_path)
                self.statusBar().showMessage("Results saved")
            except Exception as e:
                QMessageBox.warning(self, "Save Error", str(e))

    def export_plot(self):
        if not self.canvas or not self.canvas.figure:
            QMessageBox.warning(self, "Error", "No plot to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "plot.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if file_path:
            try:
                self.canvas.figure.savefig(
                    file_path, dpi=300, bbox_inches='tight')
                self.statusBar().showMessage(f"Plot saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error",
                                    f"Failed to save: {str(e)}")
