import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLineEdit, QLabel, QTableView,
                             QTextEdit, QGraphicsView, QGraphicsScene, QMessageBox,
                             QMenuBar, QAction, QStatusBar, QFileDialog, QComboBox,
                             QCheckBox, QSpinBox, QHeaderView, QDialog, QFormLayout,
                             QInputDialog, QSizePolicy)
from PyQt5.QtCore import Qt, QAbstractTableModel, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QCursor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet

from kinematics import KinematicAnalyzer


class PandasModel(QAbstractTableModel):
    """Display/edit DataFrame in QTableView."""
    data_changed = pyqtSignal()

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
                # MODIFIED: More robust type conversion
                original_dtype = self._data.dtypes[index.column()]
                if 'float' in str(original_dtype) or 'int' in str(original_dtype):
                    new_val = pd.to_numeric(value)
                else:
                    new_val = value

                self._data.iloc[index.row(), index.column()] = new_val
                self.data_changed.emit()

                self.dataChanged.emit(index, index)
                return True
            except Exception as e:
                print(f"Error setting data: {e}")
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
            return str(self._data.index[section] if hasattr(self._data, 'index') else section)
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

        # Try to guess default columns
        if 'Dip_Direction' in columns:
            self.dip_dir_combo.setCurrentText('Dip_Direction')
        elif 'DipDirection' in columns:
            self.dip_dir_combo.setCurrentText('DipDirection')

        if 'Dip_Angle' in columns:
            self.dip_combo.setCurrentText('Dip_Angle')
        elif 'Dip' in columns:
            self.dip_combo.setCurrentText('Dip')

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
        self.toolbar = None  # NEW: Store toolbar
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
        self.setWindowTitle("Kinematic Slope Analyzer (Dips-Style)")
        self.setGeometry(100, 100, 1400, 900)

        # Menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        load_act = QAction('Load Data', self)  # NEW
        load_act.triggered.connect(self.load_file)
        file_menu.addAction(load_act)
        save_res_act = QAction('Save Results', self)  # NEW
        save_res_act.triggered.connect(self.save_results_gui)
        file_menu.addAction(save_res_act)
        export_plot_act = QAction('Export Plot', self)  # NEW
        export_plot_act.triggered.connect(self.export_plot)
        file_menu.addAction(export_plot_act)

        theme_act = QAction('Toggle Dark Theme', self)
        theme_act.triggered.connect(self.toggle_theme)
        file_menu.addAction(theme_act)
        exit_act = QAction('Exit', self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # Tabs
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # --- Data Tab ---
        data_tab = QWidget()
        data_layout = QVBoxLayout()

        # Top controls
        data_ctrl_layout = QHBoxLayout()
        load_btn = QPushButton("Load Data File")
        load_btn.clicked.connect(self.load_file)
        data_ctrl_layout.addWidget(load_btn)
        map_col_btn = QPushButton("Map Dip Columns")
        map_col_btn.clicked.connect(self.map_columns)
        data_ctrl_layout.addWidget(map_col_btn)
        data_ctrl_layout.addStretch()
        data_layout.addLayout(data_ctrl_layout)

        # Manual entry
        manual_layout = QHBoxLayout()
        add_row_btn = QPushButton("Add Row")
        add_row_btn.clicked.connect(self.add_row)
        manual_layout.addWidget(add_row_btn)
        add_col_btn = QPushButton("Add Column")
        add_col_btn.clicked.connect(self.add_column)
        manual_layout.addWidget(add_col_btn)
        clear_btn = QPushButton("Clear Data")
        clear_btn.clicked.connect(self.clear_data)
        manual_layout.addWidget(clear_btn)
        manual_layout.addStretch()
        data_layout.addLayout(manual_layout)

        self.preview_table = QTableView()
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_table.setAlternatingRowColors(True)
        data_layout.addWidget(self.preview_table)

        # --- Inputs Tab (NEW) ---
        # Moved inputs to their own tab for clarity
        inputs_tab = QWidget()
        inputs_layout = QFormLayout()

        # MODIFIED: Dips tutorial default [cite: 12]
        self.slope_dir_edit = QLineEdit("135")
        # MODIFIED: Dips tutorial default [cite: 12]
        self.slope_dip_edit = QLineEdit("45")
        # MODIFIED: Dips tutorial default [cite: 59]
        self.friction_edit = QLineEdit("30")
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(1, 10)
        self.n_clusters_spin.setValue(3)
        self.lat_limit_planar_edit = QLineEdit("20")  # NEW
        self.lat_limit_topple_edit = QLineEdit("30")  # NEW [cite: 522]

        inputs_layout.addRow(QLabel("--- Slope Parameters ---"), None)
        inputs_layout.addRow(
            QLabel("Slope Dip Direction (°):"), self.slope_dir_edit)
        inputs_layout.addRow(
            QLabel("Slope Dip Angle (°):"), self.slope_dip_edit)
        inputs_layout.addRow(
            QLabel(r"Friction Angle ($\phi$) (°):"), self.friction_edit)

        inputs_layout.addRow(QLabel("--- Kinematic Limits ---"), None)
        inputs_layout.addRow(
            QLabel("Planar Lateral Limit (±°):"), self.lat_limit_planar_edit)
        inputs_layout.addRow(
            QLabel("Toppling Lateral Limit (±°):"), self.lat_limit_topple_edit)

        inputs_layout.addRow(QLabel("--- Clustering ---"), None)
        inputs_layout.addRow(
            QLabel("Number of Clusters (Sets):"), self.n_clusters_spin)

        inputs_tab.setLayout(inputs_layout)

        # --- Analysis Tab ---
        anal_tab = QWidget()
        anal_layout = QVBoxLayout()
        run_btn = QPushButton("Run Analysis")
        run_btn.clicked.connect(self.run_analysis)
        anal_layout.addWidget(run_btn)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(
            QFont("Courier New", 10))  # NEW: Monospace font
        anal_layout.addWidget(self.results_text)

        self.cluster_stats_text = QTextEdit()  # MODIFIED: Moved from old cluster tab
        self.cluster_stats_text.setReadOnly(True)
        self.cluster_stats_text.setFont(QFont("Courier New", 10))
        self.cluster_stats_text.setMaximumHeight(150)  # NEW: Set max height
        anal_layout.addWidget(QLabel("Cluster Stats:"))
        anal_layout.addWidget(self.cluster_stats_text)

        save_btn = QPushButton("Save Results CSV")
        save_btn.clicked.connect(self.save_results_gui)
        anal_layout.addWidget(save_btn)
        anal_tab.setLayout(anal_layout)

        # --- Plot Tab ---
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.plot_layout = plot_layout  # For adding canvas

        # Plot control layout
        plot_ctrl_layout = QHBoxLayout()

        # Plot type selection
        plot_type_layout = QFormLayout()
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(
            ["Stereonet", "Scatter Plot", "Bar Chart"])
        self.plot_type_combo.currentTextChanged.connect(
            self.on_plot_type_changed)
        plot_type_layout.addRow(QLabel("Plot Type:"), self.plot_type_combo)

        # NEW: Kinematic Mode dropdown
        self.failure_mode_label = QLabel("Kinematic Mode:")
        self.failure_mode_combo = QComboBox()
        self.failure_mode_combo.addItems(
            ["all_poles", "planar", "wedge", "flexural_toppling"])
        self.failure_mode_combo.currentTextChanged.connect(self.update_plot)
        plot_type_layout.addRow(self.failure_mode_label,
                                self.failure_mode_combo)

        # Projection for stereonet
        self.proj_label = QLabel("Projection:")
        self.plot_proj = QComboBox()
        self.plot_proj.addItems(["equal_area", "equal_angle"])
        self.plot_proj.currentTextChanged.connect(self.update_plot)
        plot_type_layout.addRow(self.proj_label, self.plot_proj)

        plot_ctrl_layout.addLayout(plot_type_layout)

        # Stereonet options
        self.stereo_options_layout = QVBoxLayout()
        self.show_contours_cb = QCheckBox("Show Contours")
        self.show_contours_cb.setChecked(True)
        self.show_contours_cb.stateChanged.connect(self.update_plot)
        self.stereo_options_layout.addWidget(self.show_contours_cb)

        self.show_clusters_cb = QCheckBox(
            "Show Clusters (on 'all_poles' mode)")
        self.show_clusters_cb.setChecked(True)
        self.show_clusters_cb.stateChanged.connect(self.update_plot)
        self.stereo_options_layout.addWidget(self.show_clusters_cb)

        # self.show_friction_cb = QCheckBox("Friction Cone") # MODIFIED: Removed, now part of kinematic mode
        # self.show_friction_cb.setChecked(True)
        # self.show_friction_cb.stateChanged.connect(self.update_plot)
        # self.stereo_options_layout.addWidget(self.show_friction_cb)
        self.stereo_options_layout.addStretch()
        plot_ctrl_layout.addLayout(self.stereo_options_layout)

        # Column selections (for scatter/bar)
        self.col_sel_layout = QFormLayout()
        self.col_x_label = QLabel("X Col:")
        self.x_col_combo = QComboBox()
        self.col_sel_layout.addRow(self.col_x_label, self.x_col_combo)
        self.col_y_label = QLabel("Y Col:")
        self.y_col_combo = QComboBox()
        self.col_sel_layout.addRow(self.col_y_label, self.y_col_combo)
        self.col_bar_label = QLabel("Bar Col:")
        self.bar_col_combo = QComboBox()
        self.col_sel_layout.addRow(self.col_bar_label, self.bar_col_combo)
        plot_ctrl_layout.addLayout(self.col_sel_layout)

        plot_ctrl_layout.addStretch()
        self.plot_layout.addLayout(plot_ctrl_layout)

        # Update/Export buttons
        plot_btn_layout = QHBoxLayout()
        update_plot_btn = QPushButton(
            "Generate / Update Plot")  # MODIFIED: Clearer name
        update_plot_btn.clicked.connect(self.update_plot)
        plot_btn_layout.addWidget(update_plot_btn)

        export_plot_btn = QPushButton("Export Plot")
        export_plot_btn.clicked.connect(self.export_plot)
        plot_btn_layout.addWidget(export_plot_btn)

        # MODIFIED: Clearer name
        rose_btn = QPushButton("Generate Rose Diagram (Strikes)")
        rose_btn.clicked.connect(self.plot_rose)
        plot_btn_layout.addWidget(rose_btn)
        plot_btn_layout.addStretch()
        self.plot_layout.addLayout(plot_btn_layout)

        # NEW: Canvas Container
        self.canvas_container = QWidget()
        self.canvas_layout = QVBoxLayout(self.canvas_container)
        self.canvas_container.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_layout.addWidget(self.canvas_container)

        # Add Tabs
        tabs.addTab(data_tab, "1. Data")
        tabs.addTab(inputs_tab, "2. Inputs")  # NEW
        tabs.addTab(anal_tab, "3. Analysis")
        tabs.addTab(plot_tab, "4. Plot")

        # Status Bar
        self.statusBar().showMessage("Ready")

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
        is_stereo = (plot_type == "Stereonet")
        is_scatter = (plot_type == "Scatter Plot")
        is_bar = (plot_type == "Bar Chart")

        # Stereonet options
        self.failure_mode_label.setVisible(is_stereo)
        self.failure_mode_combo.setVisible(is_stereo)
        self.proj_label.setVisible(is_stereo)
        self.plot_proj.setVisible(is_stereo)
        for i in range(self.stereo_options_layout.count()):  # MODIFIED: Loop to show/hide
            widget = self.stereo_options_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(is_stereo)

        # Scatter options
        self.col_x_label.setVisible(is_scatter)
        self.x_col_combo.setVisible(is_scatter)
        self.col_y_label.setVisible(is_scatter)
        self.y_col_combo.setVisible(is_scatter)

        # Bar options
        self.col_bar_label.setVisible(is_bar)
        self.bar_col_combo.setVisible(is_bar)

        self.update_plot()

    def apply_styles(self):
        """Light theme."""
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QTabWidget::pane { border-top: 1px solid #ccc; }
            QTabBar::tab { background: #e0e0e0; border: 1px solid #ccc; border-bottom: none; padding: 8px 16px; }
            QTabBar::tab:selected { background: #f0f0f0; margin-bottom: -1px; }
            QPushButton { background-color: #007bff; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #0056b3; }
            QLineEdit, QSpinBox, QComboBox { border: 1px solid #ccc; padding: 4px; border-radius: 2px; }
            QTextEdit { border: 1px solid #ccc; background: white; }
            QTableView { gridline-color: #ccc; background-color: white; }
            QHeaderView::section { background-color: #e8e8e8; padding: 4px; border: 1px solid #ccc; }
            QLabel { font-size: 10pt; }
            QFormLayout QLabel { font-weight: bold; }
        """)

    def toggle_theme(self):
        """Toggle dark/light."""
        current = self.styleSheet()
        if 'background-color: #f0f0f0' in current:  # Is light, switch to dark
            self.setStyleSheet("""
                QMainWindow { background-color: #2b2b2b; color: #f0f0f0; }
                QTabWidget::pane { border-top: 1px solid #444; }
                QTabBar::tab { background: #3c3c3c; border: 1px solid #444; border-bottom: none; padding: 8px 16px; color: #f0f0f0; }
                QTabBar::tab:selected { background: #2b2b2b; margin-bottom: -1px; }
                QPushButton { background-color: #007bff; color: white; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
                QPushButton:hover { background-color: #0056b3; }
                QLineEdit, QSpinBox, QComboBox { border: 1px solid #555; padding: 4px; border-radius: 2px; background-color: #3c3c3c; color: #f0f0f0; }
                QTextEdit { border: 1px solid #555; background: #3c3c3c; color: #f0f0f0; }
                QTableView { gridline-color: #555; background-color: #3c3c3c; color: #f0f0f0; }
                QHeaderView::section { background-color: #484848; padding: 4px; border: 1px solid #555; color: #f0f0f0; }
                QLabel { font-size: 10pt; color: #f0f0f0; }
                QFormLayout QLabel { font-weight: bold; }
            """)
        else:  # Is dark, switch to light
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
                    f"Loaded {len(self.analyzer.df)} rows.")
                # Auto-map columns
                self.map_columns(interactive=False)
            except Exception as e:
                QMessageBox.warning(
                    self, "Error", f"Could not load file: {str(e)}")

    def map_columns(self, interactive=True):
        if self.analyzer.df is None:
            if interactive:
                QMessageBox.warning(self, "Error", "Load data first.")
            return

        dialog = ColumnMapperDialog(self.analyzer.df.columns.tolist(), self)

        # Auto-map or show dialog
        if interactive or not (self.analyzer.dip_dir_col in self.analyzer.df.columns and self.analyzer.dip_col in self.analyzer.df.columns):
            if dialog.exec_() != QDialog.Accepted:
                return  # User cancelled

        dip_dir, dip = dialog.get_selections()
        try:
            self.analyzer.set_dip_columns(dip_dir, dip)
            self.statusBar().showMessage(
                f"Columns mapped: DipDir='{dip_dir}', Dip='{dip}'")
            # Update plot combos
            columns = self.analyzer.df.columns.tolist()
            self.x_col_combo.clear()
            self.x_col_combo.addItems(columns)
            self.y_col_combo.clear()
            self.y_col_combo.addItems(columns)
            self.bar_col_combo.clear()
            self.bar_col_combo.addItems(columns)
        except ValueError as e:
            if interactive:
                QMessageBox.warning(self, "Mapping Error", str(e))
            self.statusBar().showMessage(f"Column mapping error: {e}")

    def add_column(self):
        if self.analyzer.df is None:
            self.clear_data()  # Initialize empty dataframe

        col_name, ok = QInputDialog.getText(
            self, "Add Column", "Enter column name:")
        if ok and col_name:
            if col_name in self.analyzer.df.columns:
                QMessageBox.warning(self, "Error", "Column already exists.")
                return
            self.analyzer.df[col_name] = np.nan
            self.model.layoutChanged.emit()
            # Update combos
            self.x_col_combo.addItem(col_name)
            self.y_col_combo.addItem(col_name)
            self.bar_col_combo.addItem(col_name)
            self.statusBar().showMessage(f"Added column: {col_name}")

    def add_row(self):
        """Add empty row for manual entry."""
        if self.model is None:
            self.clear_data()  # Initialize

        new_row = pd.Series(
            [np.nan] * len(self.analyzer.df.columns), index=self.analyzer.df.columns)

        # MODIFIED: Use _data directly to ensure model updates
        self.model._data = pd.concat(
            [self.model._data, new_row.to_frame().T], ignore_index=True)
        self.analyzer.df = self.model._data

        self.model.layoutChanged.emit()
        self.preview_table.scrollToBottom()

    def clear_data(self):
        default_cols = ['Dip_Direction', 'Dip_Angle']
        self.analyzer.df = pd.DataFrame(columns=default_cols)
        self.model = PandasModel(self.analyzer.df)
        self.preview_table.setModel(self.model)
        self.model.data_changed.connect(self.on_data_changed)
        self.analyzer.set_dip_columns(
            'Dip_Direction', 'Dip_Angle')  # Set defaults
        self.statusBar().showMessage("Data cleared. Ready for manual entry.")

    def on_data_changed(self):
        """Update analyzer on edit."""
        if self.model is not None:
            self.analyzer.df = self.model._data.copy()
            try:
                # Re-validate on change
                self.analyzer._validate_data()
            except ValueError as e:
                self.statusBar().showMessage(f"Data Error: {e}")

    # MODIFIED: Removed old run_clustering, it's part of run_analysis now
    # def run_clustering(self): ...

    def run_analysis(self):
        try:
            # 1. Set parameters from GUI
            self.analyzer.set_parameters(
                float(self.slope_dir_edit.text() or 0),
                float(self.slope_dip_edit.text() or 0),
                float(self.friction_edit.text() or 0),
                self.n_clusters_spin.value(),
                lateral_limit_planar=float(
                    self.lat_limit_planar_edit.text() or 20),
                lateral_limit_toppling=float(
                    self.lat_limit_topple_edit.text() or 30)
            )

            # 2. Run analysis (this now includes clustering)
            self.analyzer.analyze()

            # 3. Update text outputs
            self.results_text.setText(self.analyzer.get_summary())
            self.cluster_stats_text.setText(self.analyzer.get_cluster_stats())

            # 4. Update plot
            self.update_plot()
            self.statusBar().showMessage("Analysis complete")
        except Exception as e:
            # NEW: Create a full, detailed error message
            error_message = f"An error occurred:\n\n"
            error_message += f"Type: {type(e).__name__}\n"
            error_message += f"Error: {str(e)}\n\n"
            error_message += "Full Traceback:\n"
            error_message += traceback.format_exc()  # This gets the full stack trace

            # Display it in a message box that can be scrolled
            QMessageBox.critical(self, "Analysis Error", error_message)
            self.statusBar().showMessage(
                f"Analysis Failed: {traceback.format_exc().splitlines()[-1]}")

    def update_plot(self):
        """Update plot based on type."""
        if self.analyzer.df is None:
            # Don't warn, just do nothing
            return

        # Clear existing canvas and toolbar
        if self.canvas:
            self.canvas_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
        if self.toolbar:
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None

        try:
            if self.current_plot_type == "Stereonet":
                if not self.analyzer.results:
                    # Don't plot if analysis hasn't run
                    return
                proj = self.plot_proj.currentText()
                mode = self.failure_mode_combo.currentText()  # NEW
                fig = self.analyzer.plot_stereonet(
                    projection=proj,
                    failure_mode=mode,  # NEW
                    show_contours=self.show_contours_cb.isChecked(),
                    show_clusters=self.show_clusters_cb.isChecked()
                )
            elif self.current_plot_type == "Scatter Plot":
                x = self.x_col_combo.currentText()
                y = self.y_col_combo.currentText()
                if not x or not y:
                    return
                fig = self.analyzer.plot_scatter(x, y)
            else:  # Bar Chart
                col = self.bar_col_combo.currentText()
                if not col:
                    return
                fig = self.analyzer.plot_bar(col)

            # Create new canvas and toolbar
            self.canvas = FigureCanvas(fig)
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.canvas_layout.addWidget(self.toolbar)  # Add toolbar
            self.canvas_layout.addWidget(self.canvas)  # Add canvas

            # No mouse move connection needed, mplstereonet handles it
            self.canvas.draw()
            self.statusBar().showMessage(
                f"Plot updated: {self.current_plot_type}")
        except Exception as e:
            QMessageBox.warning(self, "Plot Error",
                                f"Failed to generate plot: {str(e)}")

    # MODIFIED: Removed, mplstereonet has built-in hover
    # def on_mouse_move(self, event): ...

    def plot_rose(self):
        """Simple rose diagram for strikes using dip dir column."""
        if self.analyzer.df is None or self.analyzer.dip_dir_col not in self.analyzer.df.columns:
            QMessageBox.warning(self, "Error", "Load and map data first.")
            return

        try:
            # A strike is DipDir - 90
            strikes = (self.analyzer.df[self.analyzer.dip_dir_col] - 90) % 360

            # Create a separate window for the rose diagram
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='polar')

            # FIXED: Custom rose implementation (no mplstereonet.rose)
            bins = 36  # 10° bins
            bin_edges = np.linspace(0, 360, bins + 1)
            counts, _ = np.histogram(strikes, bins=bin_edges)

            # For bidirectional (strike ambiguity): Average with 180° shift
            strikes_shift = (strikes + 180) % 360
            counts_shift, _ = np.histogram(strikes_shift, bins=bin_edges)
            # Normalize to avoid double-counting
            counts = (counts + counts_shift) / 2

            # Plot polar bars
            theta = np.deg2rad(bin_edges[:-1])  # Bin centers in radians
            widths = np.deg2rad(np.diff(bin_edges))  # Bin widths in radians
            ax.bar(theta, counts, width=widths,
                   color='green', alpha=0.7, bottom=0.0)

            # Polar plot styling for rose diagrams
            ax.set_theta_zero_location('N')  # 0° at top (North)
            ax.set_theta_direction(-1)  # Clockwise rotation
            ax.set_thetagrids(np.arange(0, 360, 30), [
                              # 30° labels
                              f'{i}°' for i in np.arange(0, 360, 30)])
            ax.set_ylim(0, max(counts) * 1.1)  # Auto-scale radial limits
            ax.set_title(
                f'Strike Rose Diagram (n={len(strikes)})', y=1.08, pad=20)

            # Show in a new dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Rose Diagram")
            layout = QVBoxLayout()
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, dialog)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            dialog.setLayout(layout)
            dialog.resize(600, 600)
            dialog.exec_()

        except Exception as e:
            QMessageBox.warning(self, "Rose Plot Error", str(e))

    def save_results_gui(self):
        if not self.analyzer.results:
            QMessageBox.warning(self, "Error", "Run analysis first")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "results.csv", "CSV (*.csv)")
        if file_path:
            try:
                self.analyzer.save_results(file_path)
                self.statusBar().showMessage(f"Results saved to {file_path}")
                # Inform user about wedge file
                if self.analyzer.results['overall']['wedge_potential']:
                    QMessageBox.information(self, "Save Complete",
                                            f"Main data saved to: {file_path}\n"
                                            f"Wedge failures saved to: {file_path.replace('.csv', '_wedge_failures.csv')}")
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
