# gui/tabs/data_tab.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QTableView, QHBoxLayout,
    QFileDialog, QLabel, QLineEdit, QSpinBox, QMessageBox
)
from PySide6.QtCore import Qt, QAbstractTableModel, Signal
import pandas as pd
import numpy as np


class PandasModel(QAbstractTableModel):
    data_changed = Signal()

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return len(self._df.columns) if self._df is not None else 0

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._df is None:
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole or role == Qt.EditRole:
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            try:
                col = self._df.columns[index.column()]
                # attempt numeric conversion
                if self._df[col].dtype.kind in 'iufc':
                    self._df.iat[index.row(), index.column()] = float(value)
                else:
                    self._df.iat[index.row(), index.column()] = value
                self.data_changed.emit()
                self.layoutChanged.emit()
                return True
            except Exception:
                return False
        return False


class DataTab(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.model = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Buttons
        btn_layout = QHBoxLayout()
        load_btn = QPushButton("Load CSV")
        load_btn.clicked.connect(self.load_csv)
        add_row_btn = QPushButton("Add Row")
        add_row_btn.clicked.connect(self.add_row)
        clear_btn = QPushButton("Clear Data")
        clear_btn.clicked.connect(self.clear_data)
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(add_row_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Preview table
        self.table = QTableView()
        layout.addWidget(self.table)

        # Quick inputs
        inputs = QHBoxLayout()
        self.slope_dir = QLineEdit("125")
        self.slope_dip = QLineEdit("45")
        self.friction = QLineEdit("30")
        self.n_clusters = QSpinBox()
        self.n_clusters.setRange(1, 10)
        self.n_clusters.setValue(3)
        inputs.addWidget(QLabel("Slope Dir:"))
        inputs.addWidget(self.slope_dir)
        inputs.addWidget(QLabel("Slope Dip:"))
        inputs.addWidget(self.slope_dip)
        inputs.addWidget(QLabel("Friction:"))
        inputs.addWidget(self.friction)
        inputs.addWidget(QLabel("N Clusters:"))
        inputs.addWidget(self.n_clusters)
        layout.addLayout(inputs)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            required = ['Dip_Direction', 'Dip_Angle']
            if not all(c in df.columns for c in required):
                QMessageBox.warning(self, "Invalid CSV",
                                    f"CSV must include: {required}")
                return
            # enforce numeric types
            df['Dip_Direction'] = pd.to_numeric(
                df['Dip_Direction'], errors='coerce')
            df['Dip_Angle'] = pd.to_numeric(df['Dip_Angle'], errors='coerce')
            df = df.dropna(subset=required).reset_index(drop=True)
            self.df = df
            self.model = PandasModel(self.df)
            self.table.setModel(self.model)
            self.model.data_changed.connect(self.on_data_changed)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def add_row(self):
        if self.df is None:
            self.df = pd.DataFrame(columns=['Dip_Direction', 'Dip_Angle'])
        self.df.loc[len(self.df)] = [0.0, 0.0]
        if not self.model:
            self.model = PandasModel(self.df)
            self.table.setModel(self.model)
            self.model.data_changed.connect(self.on_data_changed)
        else:
            self.model.layoutChanged.emit()

    def clear_data(self):
        self.df = None
        self.model = None
        self.table.setModel(None)

    def on_data_changed(self):
        # keep df reference up to date (model edits already applied)
        pass
