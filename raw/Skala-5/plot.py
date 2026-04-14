#!/usr/bin/env python3
"""
ASC spectrum viewer with PySide6 + pyqtgraph (lightweight, no QtCharts/matplotlib).

- Pilih folder ASC, pilih file dari list, plot muncul (pan/zoom scroll/drag).
- Default x-range 200–900 nm.

Deps: pip install PySide6 pyqtgraph
"""
from __future__ import annotations

import sys
import csv
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None
try:
    from astropy.modeling import models, fitting
except Exception:  # noqa: BLE001
    models = None
    fitting = None
try:
    import sim as sim_module
    from sim import DataFetcher, SpectrumSimulator, MixedSpectrumSimulator
except Exception:  # noqa: BLE001
    sim_module = None
    DataFetcher = None
    SpectrumSimulator = None
    MixedSpectrumSimulator = None


def load_asc_with_optional_header(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load ASC dua kolom (λ, I), melewati baris non-float."""
    wl = []
    inten = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            wl.append(x)
            inten.append(y)
    if not wl:
        raise ValueError("Tidak ditemukan data numerik 2 kolom.")
    return np.array(wl, dtype=float), np.array(inten, dtype=float)


class SpectrumPanel(QtWidgets.QWidget):
    sigRegionChanged = QtCore.Signal(object) # Emit when ROI changes

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Top Plot (Full) ---
        self.plot_full = pg.PlotWidget()
        self.plot_full.showGrid(x=True, y=True, alpha=0.3)
        self.plot_full.setLabel("bottom", "Wavelength", units="nm")
        self.plot_full.setLabel("left", "Intensity", units="a.u.")
        self.curve_full = self.plot_full.plot([], [])
        
        # Region (ROI)
        self.region = pg.LinearRegionItem(values=(200, 900), movable=True)
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self.on_region_changed)
        self.plot_full.addItem(self.region)
        
        # --- ROI Label ---
        self.coord_label = QtWidgets.QLabel("ROI: -")
        self.coord_label.setAlignment(QtCore.Qt.AlignCenter)
        
        # --- Bottom Plot (ROI) ---
        self.plot_roi = pg.PlotWidget()
        self.plot_roi.showGrid(x=True, y=True, alpha=0.3)
        self.plot_roi.setLabel("bottom", "WL (ROI)", units="nm")
        self.plot_roi.setLabel("left", "Int", units="a.u.")
        self.plot_roi.setMouseEnabled(x=False, y=False)
        self.plot_roi.setMenuEnabled(False)
        self.curve_roi = self.plot_roi.plot([], pen=pg.mkPen(color="r", width=1.2))
        
        # Add to layout
        layout.addWidget(self.plot_full, stretch=1)
        layout.addWidget(self.coord_label)
        layout.addWidget(self.plot_roi, stretch=1)
        
        # State
        self.current_file = None
        self.data_wl = None
        self.data_inten = None
        self.data_wl_full = None
        self.data_inten_full = None
        self.peaks_markers = []
        self.fit_curve = self.plot_roi.plot([], pen=pg.mkPen('g', width=2))
        
    def load_data(self, file_path: Path, shift: float = 0.0):
        try:
            wl, inten = load_asc_with_optional_header(file_path)
            if shift != 0.0:
                wl += shift
            self.current_file = file_path
            self.data_wl_full = wl
            self.data_inten_full = inten
            
            # Simple downsampling for display speed if needed
            self.data_wl = wl[::5]
            self.data_inten = inten[::5]
            
            self.curve_full.setData(self.data_wl, self.data_inten)
            
            self.update_roi_plot()
            return True
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return False
            
    def on_region_changed(self):
        self.update_roi_plot()
        self.sigRegionChanged.emit(self)
        
    def set_region(self, r0, r1):
        self.region.blockSignals(True)
        self.region.setRegion((r0, r1))
        self.region.blockSignals(False)
        self.update_roi_plot()
        
    def update_roi_plot(self):
        if self.data_wl_full is None: return
        
        r0, r1 = self.region.getRegion()
        # Find indices
        mask = (self.data_wl_full >= r0) & (self.data_wl_full <= r1)
        
        chunk_x = self.data_wl_full[mask]
        chunk_y = self.data_inten_full[mask]
        
        if len(chunk_x) > 0:
            self.curve_roi.setData(chunk_x, chunk_y)
            self.coord_label.setText(f"ROI: {r0:.2f} - {r1:.2f} nm")
        else:
            self.curve_roi.setData([], [])

    def clear_plots(self, title: str = None):
        if title:
            self.plot_full.setTitle(title)
            self.plot_roi.setTitle(title + " (ROI)")
        self.curve_full.setData([], [])
        self.curve_roi.setData([], [])
        self.coord_label.setText("ROI: -")
        self.current_file = None
        self.data_wl_full = None
        self.data_inten_full = None




class ImportManagerDialog(QtWidgets.QDialog):
    """
    Dialog interactive untuk memilih puncak dari template.
    Fitur:
    - Group by Element
    - Checkbox selection
    - Click to Navigate (Preview)
    """
    def __init__(self, parent, df_src, main_window):
        super().__init__(parent)
        self.setWindowTitle("Import Peaks Manager")
        self.resize(600, 700)
        self.df_src = df_src
        self.main_window = main_window # Reference to PlotViewer for navigation
        self.setup_ui()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        lbl_info = QtWidgets.QLabel(f"Ditemukan {len(self.df_src)} puncak di template.\n"
                                  "Centang puncak yang ingin di-import. Klik baris untuk preview.")
        layout.addWidget(lbl_info)
        
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Select", "Element", "Wavelength (nm)", "Ion", "Status"])
        self.tree.setColumnWidth(0, 80)
        self.tree.setColumnWidth(1, 80)
        layout.addWidget(self.tree)
        
        # Populate
        groups = {}
        for idx, row in self.df_src.iterrows():
            el = str(row.get('element', '?')).strip()
            if not el or el == 'nan': el = 'Unknown'
            if el not in groups: groups[el] = []
            groups[el].append(row)
            
        for el in sorted(groups.keys()):
            # Parent Item (Element Header)
            parent = QtWidgets.QTreeWidgetItem([f"", el, "", "", ""])
            parent.setFlags(parent.flags() | QtCore.Qt.ItemIsAutoTristate | QtCore.Qt.ItemIsUserCheckable)
            parent.setCheckState(0, QtCore.Qt.Checked)
            self.tree.addTopLevelItem(parent)
            
            for row in groups[el]:
                try:
                    wl = float(row.get('center_nm', 0))
                except: wl = 0
                ion = str(row.get('ion_or_sp', ''))
                desc = f"{wl:.3f}"
                
                child = QtWidgets.QTreeWidgetItem(["", el, desc, ion, "Pending"])
                child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                child.setCheckState(0, QtCore.Qt.Checked)
                # Store row data via UserRole
                child.setData(0, QtCore.Qt.UserRole, row.to_dict())
                parent.addChild(child)
                
        self.tree.expandAll()
        self.tree.itemClicked.connect(self.on_item_clicked)
        
        btn_box = QtWidgets.QHBoxLayout()
        
        btn_all = QtWidgets.QPushButton("Select All")
        btn_all.clicked.connect(lambda: self.set_all_checked(True))
        btn_none = QtWidgets.QPushButton("Select None")
        btn_none.clicked.connect(lambda: self.set_all_checked(False))
        
        self.btn_import = QtWidgets.QPushButton("Process Import")
        self.btn_import.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        self.btn_import.clicked.connect(self.accept)
        
        btn_box.addWidget(btn_all)
        btn_box.addWidget(btn_none)
        btn_box.addStretch()
        btn_box.addWidget(self.btn_import)
        
        layout.addLayout(btn_box)
        
    def set_all_checked(self, checked):
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            parent = root.child(i)
            parent.setCheckState(0, state)
            # Tristate logic handles children usually, but explicit loop is safer if needed
            
    def on_item_clicked(self, item, col):
        data = item.data(0, QtCore.Qt.UserRole)
        if data:
            try:
                wl = float(data.get('center_nm', 0))
                if wl > 50: # Valid wavelength
                    self.main_window.navigate_to_wavelength(wl)
            except: pass
                
    def get_selected_rows(self):
        selected_rows = []
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            parent = root.child(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == QtCore.Qt.Checked:
                    data = child.data(0, QtCore.Qt.UserRole)
                    if data: selected_rows.append(data)
        return selected_rows

class PlotViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASC Spectrum Viewer (pyqtgraph)")
        self.resize(1200, 750)

        self.data_wl: Optional[np.ndarray] = None          # downsampled for display
        self.data_inten: Optional[np.ndarray] = None       # downsampled for display
        self.data_wl_full: Optional[np.ndarray] = None     # full-resolution for fitting
        self.data_inten_full: Optional[np.ndarray] = None  # full-resolution for fitting
        self.current_file: Optional[Path] = None
        self.current_folder: Optional[Path] = None
        self.move_sensitivity = 1.0
        self.use_full_display = False
        self.skip_revert_once = False
        self.first_load_done = False

        folder_btn = QtWidgets.QPushButton("Pilih Folder ASC...")
        folder_btn.clicked.connect(self.choose_folder)

        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.list_files = QtWidgets.QListWidget()
        self.list_files.itemSelectionChanged.connect(self.load_selected_file)

        self.fit_btn = QtWidgets.QPushButton("Fit Voigt (ROI)")
        self.fit_btn.clicked.connect(self.fit_voigt_roi)
        self.fit_btn.setEnabled(models is not None)
        self.export_enabled = True
        self.export_btn = QtWidgets.QPushButton("Ekspor: ON")
        self.export_btn.setCheckable(True)
        self.export_btn.setChecked(True)
        self.export_btn.clicked.connect(self.toggle_export)
        self.sim_btn = QtWidgets.QPushButton("Overlay Simulasi")
        self.sim_btn.setEnabled(DataFetcher is not None and MixedSpectrumSimulator is not None)
        self.sim_btn.clicked.connect(self.run_sim_overlay)
        self.sim_overlay_cb = QtWidgets.QCheckBox("Overlay ON")
        self.sim_overlay_cb.setChecked(True)
        self.sim_overlay_cb.stateChanged.connect(self.toggle_sim_overlay)
        self.sim_label_cb = QtWidgets.QCheckBox("Label Sim Peaks (ROI)")
        self.sim_label_cb.stateChanged.connect(self.update_roi_from_region)
        self.save_state_btn = QtWidgets.QPushButton("Simpan sesi")
        self.save_state_btn.clicked.connect(lambda: self.save_state(prompt_if_missing=True))
        self.load_state_btn = QtWidgets.QPushButton("Load sesi")
        self.load_state_btn.clicked.connect(self.load_state)
        self.last_session_path: Optional[Path] = None
        self.full_view_cb = QtWidgets.QCheckBox("Tampil full-res (sementara)")
        self.full_view_cb.setChecked(False)
        self.full_view_cb.stateChanged.connect(self.toggle_full_display)
        self.detect_btn = QtWidgets.QPushButton("Deteksi puncak")
        self.detect_btn.clicked.connect(lambda: self.update_peak_detection(auto_center=False))
        self.prev_peak_btn = QtWidgets.QPushButton("<<")
        self.prev_peak_btn.clicked.connect(lambda: self.jump_peak(-1))
        self.next_peak_btn = QtWidgets.QPushButton(">>")
        self.next_peak_btn.clicked.connect(lambda: self.jump_peak(1))
        self.prev_peak_btn.setEnabled(False)
        self.next_peak_btn.setEnabled(False)
        self.collect_btn = QtWidgets.QPushButton("Simpan puncak")
        self.collect_btn.setEnabled(False)
        self.collect_btn.clicked.connect(self.save_peak_record)
        self.collection_count = 0
        self.collection_data: List[dict] = []
        self.collection_data: List[dict] = []
        self.collection_data_by_file: dict[str, List[dict]] = {}
        # Cache for per-file metadata (e.g. Te_K) to avoid re-reading Summary CSV
        self.file_metadata_cache: dict[str, dict] = {}
        self.collection_table = QtWidgets.QTableWidget(0, 7)
        self.collection_table.setHorizontalHeaderLabels(
            ["File", "Center (nm)", "Peak", "Area", "FWHM", "ROI", "PNG"]
        )
        self.collection_table.horizontalHeader().setStretchLastSection(True)
        self.collection_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.collection_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.collection_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.collection_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.collection_table.itemDoubleClicked.connect(self.on_collection_double_click)
        self.collection_table.itemDoubleClicked.connect(self.on_collection_double_click)
        # self.collection_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu) 
        # Context menu removed in favor of explicit delete button below
        # unified lines table (target + koleksi)
        # unified lines table (target + koleksi)
        self.target_lines: List[Dict] = []  # Changed to list of dicts with full atomic data
        self.lines_table = QtWidgets.QTableWidget(0, 13)
        self.lines_table.setHorizontalHeaderLabels(
            ["Element", "Ion", "λ (nm)", "Aki", "Ei (eV)", "Ek (eV)", "File", "Center fit", "Peak", "Area", "ROI", "Status", "Δλ"]
        )
        self.lines_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.lines_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.lines_table.horizontalHeader().setStretchLastSection(True)
        self.lines_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.lines_table.verticalHeader().setVisible(False)

        # Remove explicit Import/Export buttons from UI if they confuse workflow?
        # User wants auto-update. Keeping them as manual fallback is okay, but purely manual.
        self.load_session_btn = QtWidgets.QPushButton("Reload File State")
        self.load_session_btn.clicked.connect(self.reload_current_file_state)
        
        self.calibrate_btn = QtWidgets.QPushButton("Kalibrasi Shift")
        self.calibrate_btn.setToolTip("Geser sumbu X (wavelength) untuk koreksi shift")
        self.calibrate_btn.clicked.connect(self.calibrate_shift_dialog)
        self.calibrate_btn.setStyleSheet("background-color: #E3F2FD; color: #0D47A1;")
        
        self.delete_line_btn = QtWidgets.QPushButton("Hapus Puncak Terpilih")
        self.delete_line_btn.clicked.connect(self.remove_peak_from_selection)
        self.delete_line_btn.setStyleSheet("background-color: #ffdddd; color: red;")

        self.clear_session_btn = QtWidgets.QPushButton("Hapus koleksi")
        self.clear_session_btn.clicked.connect(self.clear_collection_session)
        self.min_amp_edit = QtWidgets.QLineEdit()
        self.min_amp_edit.setPlaceholderText("Min amp deteksi")
        self.min_amp_edit.setFixedWidth(90)
        self.min_dist_edit = QtWidgets.QLineEdit()
        self.min_dist_edit.setPlaceholderText("Min jarak nm")
        self.min_dist_edit.setFixedWidth(90)
        self.prom_edit = QtWidgets.QLineEdit()
        self.prom_edit.setPlaceholderText("Prom % max")
        self.prom_edit.setFixedWidth(90)
        self.max_peaks_edit = QtWidgets.QLineEdit()
        self.max_peaks_edit.setPlaceholderText("Max peaks")
        self.max_peaks_edit.setFixedWidth(80)
        self.max_peaks_edit.setText("5")
        self.start_peak_edit = QtWidgets.QLineEdit()
        self.start_peak_edit.setPlaceholderText("Start λ nm")
        self.start_peak_edit.setFixedWidth(90)
        self.lock_width_cb = QtWidgets.QCheckBox("Lock ROI width")
        self.lock_width_cb.setChecked(False)
        self.last_roi_width = None
        self.last_fit_result = None
        self.detected_peaks: List[Tuple[float, float]] = []
        self.peak_index: int = -1
        self.peak_marker_full = None
        self.peak_marker_roi = None
        self.last_selected_peaks: List[Tuple[float, float]] = []
        # Calibration Offsets (Filename -> Shift in nm)
        self.calibration_offsets: Dict[str, float] = {}
        # Persistent calibration reference state
        self.last_calib_element: str = "Ca"
        self.last_calib_ion: str = "I"
        self.last_calib_aki: str = "1e8"
        self.last_calib_ref_wl: Optional[float] = None
        # load daftar elemen dari HDF untuk auto-complete
        self.available_elements = self._load_elements_from_hdf("nist_data_hog_augmented.h5")
        completer = QtWidgets.QCompleter(self.available_elements)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        # input overlay di header
        # NIST Database Controls
        self.nist_elem_edit = QtWidgets.QLineEdit("Fe")
        self.nist_elem_edit.setPlaceholderText("El (e.g. Fe)")
        self.nist_elem_edit.setFixedWidth(60)
        self.nist_elem_edit.setCompleter(completer)
        
        self.nist_ion_combo = QtWidgets.QComboBox()
        self.nist_ion_combo.addItems(["I", "II", "III", "All"])
        self.nist_ion_combo.setFixedWidth(60)
        
        self.nist_min_aki = QtWidgets.QLineEdit("1e5")
        self.nist_min_aki.setPlaceholderText("Min Aki")
        self.nist_min_aki.setFixedWidth(70)
        
        self.nist_min_aki = QtWidgets.QLineEdit("1e5")
        self.nist_min_aki.setPlaceholderText("Min Aki")
        self.nist_min_aki.setFixedWidth(70)
        
        self.fetch_nist_btn = QtWidgets.QPushButton("Fetch NIST")
        self.fetch_nist_btn.clicked.connect(self.fetch_nist_lines)
        
        # self.save_targets_btn removed as per user request


        self.comp_edit = QtWidgets.QLineEdit("Fe, Al, Si")
        self.comp_edit.setPlaceholderText("Atom, pisah koma")
        self.comp_edit.setFixedWidth(120)
        self.comp_edit.setCompleter(completer)
        # self.comp_edit.textChanged.connect(lambda _=None: self.refresh_lines_table()) # Removed automatic refresh on text change for now

        self.temp_edit = QtWidgets.QLineEdit("12000")
        self.temp_edit.setFixedWidth(70)
        self.ne_edit = QtWidgets.QLineEdit("1e17")
        self.ne_edit.setFixedWidth(70)
        self.fwhm_line_edit = QtWidgets.QLineEdit()
        self.fwhm_line_edit.setPlaceholderText("FWHM garis nm")
        self.fwhm_line_edit.setFixedWidth(90)
        self.fwhm_conv_edit = QtWidgets.QLineEdit()
        self.fwhm_conv_edit.setPlaceholderText("FWHM konv nm")
        self.fwhm_conv_edit.setFixedWidth(95)
        # default isi dari SIMULATION_CONFIG jika ada
        try:
            from sim import SIMULATION_CONFIG as _SC  # type: ignore
            if "sigma" in _SC:
                self.fwhm_line_edit.setText(f"{float(_SC['sigma']) * 2.355:.4f}")
            if "convolution_sigma" in _SC:
                self.fwhm_conv_edit.setText(f"{float(_SC['convolution_sigma']) * 2.355:.4f}")
        except Exception:
            pass

        # pyqtgraph plots: overview + ROI
        pg.setConfigOptions(antialias=True)
        # Multi-Spectrum Panels (3 Panels)
        self.panels = [SpectrumPanel() for _ in range(3)]
        
        # Alias for backward compatibility (Primary Panel = Panel 0)
        self.plot_full = self.panels[0].plot_full
        self.curve_full = self.panels[0].curve_full
        self.region = self.panels[0].region
        self.coord_label = self.panels[0].coord_label
        self.plot_roi = self.panels[0].plot_roi
        self.curve_roi = self.panels[0].curve_roi
        # Connect Sync
        for panel in self.panels:
            panel.sigRegionChanged.connect(self.sync_roi_regions)
            
        self.sim_label_items: List[pg.TextItem] = []

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(QtWidgets.QLabel("Files:"))
        left_layout.addWidget(self.list_files, stretch=1)
        
        # NIST Controls Layout
        nist_layout = QtWidgets.QHBoxLayout()
        nist_layout.addWidget(QtWidgets.QLabel("NIST DB:"))
        nist_layout.addWidget(self.nist_elem_edit)
        nist_layout.addWidget(self.nist_ion_combo)
        nist_layout.addWidget(self.nist_min_aki)
        nist_layout.addWidget(self.fetch_nist_btn)
        # nist_layout.addWidget(self.save_targets_btn)
        nist_layout.addStretch()
        left_layout.addLayout(nist_layout)

        
        # panel lines gabungan
        # panel lines gabungan
        lines_header_layout = QtWidgets.QHBoxLayout()
        lines_header_layout.addWidget(QtWidgets.QLabel("Target Lines (NIST) + Koleksi Match:"))
        lines_header_layout.addStretch()
        lines_header_layout.addWidget(self.delete_line_btn)
        
        left_layout.addLayout(lines_header_layout)
        left_layout.addWidget(self.lines_table, stretch=1)
        self.lines_table.itemDoubleClicked.connect(self.on_lines_table_double_click)
        # tombol import/export koleksi disembunyikan karena sesi sudah terintegrasi
        # self.load_session_btn removed
        # self.save_session_btn removed

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_layout)

        right_layout = QtWidgets.QVBoxLayout()
        
        panels_layout = QtWidgets.QHBoxLayout()
        for p in self.panels:
            panels_layout.addWidget(p)
            
        right_layout.addLayout(panels_layout, stretch=10)
        self.result_view = QtWidgets.QTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setPlaceholderText("Fit results (total + per peak)...")
        self.result_view.setMinimumHeight(140)
        right_layout.addWidget(self.result_view)
        
        # Boltzmann Diagnostic Button
        self.boltzmann_btn = QtWidgets.QPushButton("Cek Boltzmann Plot (Diagnostik Self-Absorption)")
        self.boltzmann_btn.clicked.connect(self.show_boltzmann_plot)
        right_layout.addWidget(self.boltzmann_btn)
        
        self.btn_boltz_export = QtWidgets.QPushButton("Diagnosis SA (Export)")
        self.btn_boltz_export.clicked.connect(self.export_boltzmann_diagnosis)
        right_layout.addWidget(self.btn_boltz_export)
        
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_layout)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

        toolbar_inner = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(toolbar_inner)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        def section_label(text: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(text)
            font = lbl.font()
            font.setBold(True)
            lbl.setFont(font)
            return lbl

        # Data/Export row
        data_row = QtWidgets.QHBoxLayout()
        data_row.addWidget(section_label("Data:"))
        data_row.addWidget(section_label("Data:"))
        data_row.addWidget(folder_btn)
        data_row.addWidget(self.export_btn)
        data_row.addWidget(self.save_state_btn)
        data_row.addWidget(self.load_state_btn)
        data_row.addWidget(self.calibrate_btn)
        self.import_peaks_btn = QtWidgets.QPushButton("Import") # Short text
        self.import_peaks_btn.setToolTip("Import Puncak dari Sampel Lain (Ctrl+I)")
        self.import_peaks_btn.clicked.connect(self.import_peaks_dialog)
        data_row.addWidget(self.import_peaks_btn)
        
        self.batch_automate_btn = QtWidgets.QPushButton("Batch MulFit")
        self.batch_automate_btn.setToolTip("Proses otomatis 3 iterasi multi-fit dari folder 0 ke 0-b")
        self.batch_automate_btn.clicked.connect(self.batch_automate_iterations)
        self.batch_automate_btn.setStyleSheet("background-color: #FFECB3; color: #E65100; font-weight: bold;")
        data_row.addWidget(self.batch_automate_btn)
        
        # Reload Button (Sync Deletion)
        self.reload_file_btn = QtWidgets.QPushButton("Reload DB")
        self.reload_file_btn.setToolTip("Reload Collection dari Disk (Gunakan setelah menghapus permanent di CFL)")
        self.reload_file_btn.clicked.connect(self.reload_current_file_state)
        data_row.addWidget(self.reload_file_btn)
        
        # Buka CFL Analyzer Button
        self.btn_open_cfl = QtWidgets.QPushButton("Buka CFL Analyzer")
        self.btn_open_cfl.setToolTip("Buka Jendela Analisis CF-LIBS")
        self.btn_open_cfl.clicked.connect(self.open_cfl_analyzer)
        data_row.addWidget(self.btn_open_cfl)
        
        data_row.addWidget(self.full_view_cb)
        data_row.addStretch(1)
        top_layout.addLayout(data_row)

        # Deteksi/Fit row
        det_row = QtWidgets.QHBoxLayout()
        det_row.addWidget(section_label("Deteksi/Fit:"))
        det_row.addWidget(self.detect_btn)
        det_row.addWidget(self.prev_peak_btn)
        det_row.addWidget(self.next_peak_btn)
        det_row.addWidget(self.fit_btn)
        det_row.addWidget(self.collect_btn)
        
        self.multi_peak_cb = QtWidgets.QCheckBox("Multi-Fit")
        self.multi_peak_cb.setToolTip("Fit semua puncak yang terdeteksi di ROI (Deconvolution)")
        det_row.addWidget(self.multi_peak_cb)
        
        det_row.addWidget(QtWidgets.QLabel("Min amp"))
        det_row.addWidget(self.min_amp_edit)
        det_row.addWidget(QtWidgets.QLabel("Min jarak"))
        det_row.addWidget(self.min_dist_edit)
        det_row.addWidget(QtWidgets.QLabel("Prom %"))
        det_row.addWidget(self.prom_edit)
        det_row.addWidget(QtWidgets.QLabel("Max"))
        det_row.addWidget(self.max_peaks_edit)
        det_row.addWidget(QtWidgets.QLabel("Start λ"))
        det_row.addWidget(self.start_peak_edit)
        det_row.addWidget(self.lock_width_cb)
        det_row.addStretch(1)
        top_layout.addLayout(det_row)

        # Simulasi row
        sim_row = QtWidgets.QHBoxLayout()
        sim_row.addWidget(section_label("Simulasi:"))
        sim_row.addWidget(self.sim_btn)
        sim_row.addWidget(self.sim_overlay_cb)
        sim_row.addWidget(self.sim_label_cb)
        sim_row.addWidget(QtWidgets.QLabel("Atom"))
        sim_row.addWidget(self.comp_edit)
        sim_row.addWidget(QtWidgets.QLabel("T(K)"))
        sim_row.addWidget(self.temp_edit)
        sim_row.addWidget(QtWidgets.QLabel("n_e"))
        sim_row.addWidget(self.ne_edit)
        sim_row.addWidget(QtWidgets.QLabel("FWHM garis"))
        sim_row.addWidget(self.fwhm_line_edit)
        sim_row.addWidget(QtWidgets.QLabel("FWHM konv"))
        sim_row.addWidget(self.fwhm_conv_edit)
        sim_row.addStretch(1)
        top_layout.addLayout(sim_row)

        toolbar_inner.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        toolbar_scroll = QtWidgets.QScrollArea()
        toolbar_scroll.setWidget(toolbar_inner)
        toolbar_scroll.setWidgetResizable(True)
        toolbar_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        toolbar_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        toolbar_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        toolbar_scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(toolbar_scroll)
        main_layout.addWidget(self.file_label)
        main_layout.addWidget(splitter, stretch=1)

        container = QtWidgets.QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        # mouse move proxy untuk koordinat pada plot full
        self.proxy_mouse = pg.SignalProxy(self.plot_roi.scene().sigMouseMoved, rateLimit=30, slot=self.on_mouse_moved_roi)

    def choose_folder(self):
        default_dir = str((Path.cwd() / "0") if (Path.cwd() / "0").exists() else Path.cwd())
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Pilih Folder ASC", default_dir)
        if not path:
            return
        self.current_folder = Path(path)
        self.refresh_file_list()

    def load_selected_file(self):
        if self.current_folder is None:
            return
        items = self.list_files.selectedItems()
        if not items:
            return
            
        # Determine 3 files
        row = self.list_files.row(items[0])
        total = self.list_files.count()
        
        # Panel 0 (Primary)
        fname0 = items[0].text()
        path0 = self.current_folder / fname0
        self.load_file_data(path0)
        
        # Panel 1 & 2
        for offset in [1, 2]:
            idx = row + offset
            panel = self.panels[offset]
            if idx < total:
                fname = self.list_files.item(idx).text()
                shift = self.calibration_offsets.get(fname, 0.0)
                panel.load_data(self.current_folder / fname, shift=shift)
                if shift != 0.0:
                    print(f"[CALIBRATION] Applied shift {shift:+.4f} nm to {fname}")
            else:
                panel.clear_plots()
        
        # Update display (Titles, Axes) for all panels now that they are loaded
        self.render_plot()
                
    def sync_roi_regions(self, source_panel):
        """Sync ROI of all panels to match source_panel."""
        r0, r1 = source_panel.region.getRegion()
        for p in self.panels:
            if p is not source_panel:
                p.set_region(r0, r1)

    def load_file_data(self, path: Path):
        """Load data from path and update UI state."""
        try:
            shift = self.calibration_offsets.get(path.name, 0.0)
            if self.panels:
                self.panels[0].load_data(path, shift=shift)
                if shift != 0.0:
                    print(f"[CALIBRATION] Applied shift {shift:+.4f} nm to {path.name}")
                self.data_wl_full = self.panels[0].data_wl_full
                self.data_inten_full = self.panels[0].data_inten_full
            else:
                wl, inten = load_asc_with_optional_header(path)
                if shift != 0.0:
                    wl += shift
                self.data_wl_full, self.data_inten_full = wl, inten
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
            return
            
        # simpan full-res untuk fitting, tampilkan versi downsample agar plot ringan
        # data_wl_full/inten_full must be available now
        
        # mode tampilan: default full-res hanya untuk load pertama; ganti sampel → kembali low-res
        if not self.first_load_done and self.full_view_cb.isChecked():
            self.use_full_display = True
        else:
            if self.full_view_cb.isChecked():
                self.full_view_cb.blockSignals(True)
                self.full_view_cb.setChecked(False)
                self.full_view_cb.blockSignals(False)
            self.use_full_display = False
        self.skip_revert_once = False
        self.apply_display_mode()
        self.current_file = path
        # set koleksi sesuai file (per-sample)
        key = self.current_file.name
        if key not in self.collection_data_by_file:
            self.collection_data_by_file[key] = []
        self.collection_data = self.collection_data_by_file[key]
        self.collection_count = len(self.collection_data)
        self.last_fit_result = None
        self.collect_btn.setEnabled(False)
        self.file_label.setText(f"{path.name} | λ range {self.data_wl_full.min():.3f}-{self.data_wl_full.max():.3f} nm")
        self.try_load_temp_from_csv(path.name)
        self.render_plot()
        self.reload_current_file_state() # RESTORED: Load saved peaks from Excel
        self.update_peak_detection(auto_center=False)
        # self.refresh_collection_table() # Will be called by caller if needed, but safe here too
        # self.refresh_lines_table()
        # self.first_load_done = True
        
        # Note: The original logic included these updates. 
        # But 'reload_current_file_state' calls its own refresh.
        # Let's keep them here to ensure 'load_selected_file' still works identically.
        self.update_peak_detection(auto_center=False)
        self.refresh_collection_table()
        self.refresh_lines_table()
        self.first_load_done = True


    def clear_plot(self, title: str):
        self.curve_full.setData([], [])
        self.curve_roi.setData([], [])
        self.plot_full.setTitle(title)
        self.plot_roi.setTitle(title + " (ROI)")
        self.plot_full.setXRange(0, 1)
        self.plot_full.setYRange(0, 1)
        self.plot_roi.setXRange(0, 1)
        self.plot_roi.setYRange(0, 1)

    def render_plot(self):
        """Redraw all panels based on current data and display settings."""
        if not self.panels: return

        # Sync region bounds to fit longest data if needed?
        # Or just use the first valid panel as "Master" for bounds?
        # Let's find global Min/Max x to set bounds if needed, or per-panel.
        # But panels share ROI X-axis usually.
        
        # 1. Update Data (Downsampling) & Full Plot
        for idx, panel in enumerate(self.panels):
            if panel.current_file is None:
                panel.clear_plots(f"Panel {idx+1}: Empty")
                continue
                
            # Apply Downsampling
            if self.use_full_display:
                panel.data_wl = panel.data_wl_full
                panel.data_inten = panel.data_inten_full
            else:
                panel.data_wl, panel.data_inten = self.downsample(panel.data_wl_full, panel.data_inten_full)
            
            # Update Full Curve
            panel.curve_full.setData(panel.data_wl, panel.data_inten)
            panel.curve_full.setDownsampling(auto=not self.use_full_display)
            panel.curve_full.setClipToView(not self.use_full_display)
            
            # Set View Range and Padding
            xmin = float(panel.data_wl.min())
            xmax = float(panel.data_wl.max())
            panel.plot_full.setXRange(xmin, xmax, padding=0)
            
            ymin = float(panel.data_inten.min())
            ymax = float(panel.data_inten.max())
            if ymin == ymax: ymax += 1.0
            panel.plot_full.setYRange(min(ymin, 0.0), ymax * 1.1, padding=0)
            
            # Title
            t_base = panel.current_file.name
            t_suf = " [full-res]" if self.use_full_display else ""
            panel.plot_full.setTitle(f"{t_base}{t_suf}")
            panel.plot_roi.setTitle(f"{t_base}{t_suf} (ROI)")
            
            # Update ROI Content
            # We call update_roi_plot to redraw the ROI curve and handle sim overlay
            self.update_panel_roi(panel)

        # Update region bounds logic?
        # self.check_region_bounds()? 
        # For now assume sync_roi_regions handles positions.

    def update_panel_roi(self, panel):
        """Update ROI curve and Overlay for a specific panel."""
        r0, r1 = panel.region.getRegion()
        
        # Enforce minimum width logic (legacy from update_roi_from_region)
        width = r1 - r0
        if self.lock_width_cb.isChecked():
            if self.last_roi_width is None: self.last_roi_width = width if width > 0 else 0.2
            target_w = self.last_roi_width
            mid = (r0 + r1) / 2.0
            r0, r1 = mid - target_w/2.0, mid + target_w/2.0
            if r1 != panel.region.getRegion()[1]:
                 panel.region.setRegion((r0, r1)) # triggering signal? block?
                 # blocking handled in sync usually?
        else:
            self.last_roi_width = width
            
        # Draw ROI Data
        if panel.data_wl is None: return
        mask = (panel.data_wl >= r0) & (panel.data_wl <= r1)
        
        if np.any(mask):
            x_roi = panel.data_wl[mask]
            y_roi = panel.data_inten[mask]
            # Finite check
            valid = np.isfinite(x_roi) & np.isfinite(y_roi)
            if np.any(valid):
                panel.curve_roi.setData(x_roi[valid], y_roi[valid])
            else:
                 panel.curve_roi.setData([], [])
        else:
            panel.curve_roi.setData([], [])
            
        # Sim Overlay
        if hasattr(self, "sim_data") and self.sim_data and self.sim_overlay_cb.isChecked():
            sim_wl, sim_spec = self.sim_data
            # Mask sim to ROI
            s_mask = (sim_wl >= r0) & (sim_wl <= r1)
            # If nothing in ROI, maybe show full? Legacy behavior: "menampilkan seluruh"
            if not np.any(s_mask):
                s_mask = np.ones_like(sim_wl, dtype=bool)
                
            sim_sub = sim_spec[s_mask]
            sim_x = sim_wl[s_mask]
            
            # Scale Overlay to match Panel Data Peak in ROI
            scale_target = 1.0
            if np.any(mask):
                # max of data in roi
                scale_target = float(np.nanmax(panel.data_inten[mask]))
            else:
                scale_target = float(np.nanmax(panel.data_inten)) if panel.data_inten is not None else 1.0
                
            sim_max = float(np.nanmax(sim_sub)) if sim_sub.size else 0.0
            if sim_max > 0:
                ratio = scale_target / sim_max
                sim_sub = sim_sub * min(ratio, 1e6)
            
            # Ensure panel has sim_curve check?
            if not hasattr(panel, "sim_curve_roi"):
                 # Create it if missing
                 panel.sim_curve_roi = panel.plot_roi.plot([], pen=pg.mkPen(color="orange", width=2))
                 
            panel.sim_curve_roi.setData(sim_x, sim_sub)
        else:
            if hasattr(panel, "sim_curve_roi"):
                 panel.sim_curve_roi.setData([], [])

    def apply_display_mode(self, force_downsample: bool = False):
        if force_downsample:
            self.use_full_display = False
            self.skip_revert_once = False
            self.full_view_cb.blockSignals(True)
            self.full_view_cb.setChecked(False)
            self.full_view_cb.blockSignals(False)
        # Data update happens in render_plot now
        
    def toggle_full_display(self):
        self.use_full_display = self.full_view_cb.isChecked()
        self.skip_revert_once = self.use_full_display
        self.render_plot()
        # self.update_peak_detection(auto_center=False) # Should update peaks on Panel 0?
        # Peak detection is mainly for "Finding" peaks, usually manual or semi-auto.
        # We can trigger it if needed but maybe optional.
        if self.use_full_display:
            val = len(self.panels[0].data_wl_full) if (self.panels and self.panels[0].data_wl_full is not None) else 0
            self.statusBar().showMessage(f"Full-res ({val} pts)", 3000)
        else:
            self.statusBar().showMessage("Downsampled", 3000)

    # Legacy method shim if needed or just removed
    def update_roi_from_region(self):
        # Redirect to panel 0 update for compatibility if called externally?
        if self.panels:
            self.update_panel_roi(self.panels[0])
    # -------- Helpers --------
    def _find_peaks(
        self,
        x: np.ndarray,
        y: np.ndarray,
        min_amp: Optional[float] = None,
        min_dist: Optional[float] = None,
        rel_prom: Optional[float] = None,
        max_peaks: Optional[int] = None,
    ) -> List[Tuple[float, float]]:
        """Cari puncak sederhana dengan threshold, jarak minimal, dan prominence opsional."""
        peaks = []
        if x.size < 3:
            return peaks
        ymax = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
        prom_thresh = None
        if rel_prom is not None and rel_prom > 0 and ymax > 0:
            prom_thresh = rel_prom * ymax
        for i in range(1, x.size - 1):
            if y[i] > 0 and y[i] > y[i - 1] and y[i] > y[i + 1]:
                if min_amp is not None and y[i] < min_amp:
                    continue
                if prom_thresh is not None and y[i] < prom_thresh:
                    continue
                peaks.append((float(x[i]), float(y[i])))
        # urutkan puncak dari tinggi terbesar
        peaks.sort(key=lambda t: t[1], reverse=True)
        if min_dist is not None and min_dist > 0:
            filtered = []
            for px, py in peaks:
                if all(abs(px - fx) >= min_dist for fx, _ in filtered):
                    filtered.append((px, py))
            peaks = filtered
        if max_peaks is not None and max_peaks > 0:
            peaks = peaks[:max_peaks]
        return peaks

    def _get_fit_message(self, fitter) -> str:
        """Ambil pesan fit_info dari fitter."""
        try:
            msg = fitter.fit_info.get("message", "")
            if isinstance(msg, (list, tuple)):
                msg = " ".join(str(m) for m in msg)
            return str(msg)
        except Exception:
            return ""

    def _choose_peaks(self, peaks: List[Tuple[float, float]], silent: bool = False, target_center: float = None, context_info: str = None, preview_callback=None, multi_fit=False) -> List[Tuple[float, float]]:
        """Dialog pilih satu atau banyak puncak. Silent: pilih terdekat dgn target_center (jika ada) atau max amp.
        Preview Callback: Fungsi(selection) -> None, dipanggil saat seleksi berubah untuk update plot.
        """
        if silent:
            if not peaks: return []
            
            # Sort by intensity descending default
            peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
            
            if multi_fit:
                 # Return ALL detected peaks (that passed detection thresholds)
                 # We sort by wavelength to have ordered components usually? 
                 # Or keep default detection order (likely by WL).
                 # Let's sort by WL for predictable component indexing.
                 return sorted(peaks, key=lambda x: x[0])

            # Prioritas: Terdekat dengan target_center (tengah ROI)
            if target_center is not None:
                # Sort by distance to center
                peaks_sorted = sorted(peaks, key=lambda x: abs(x[0] - target_center))
                return [peaks_sorted[0]]
            else:
                # Sort by intensity
                peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
                return [peaks_sorted[0]]
                
        peaks_list = list(peaks)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Pilih puncak untuk fit")
        
        # Position logic: Top Left relative to main window
        geo = self.geometry()
        x_pos = geo.x() + 50
        y_pos = geo.y() + 100
        dialog.move(x_pos, y_pos)
        
        layout = QtWidgets.QVBoxLayout(dialog)

        
        # Add Context Info Label (NEW)
        if context_info:
            lbl_ctx = QtWidgets.QLabel(context_info)
            lbl_ctx.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
            layout.addWidget(lbl_ctx)

        info = QtWidgets.QLabel("Pilih satu atau lebih puncak (bisa tambah λ manual):")
        layout.addWidget(info)
        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for px, py in peaks_list:
            listw.addItem(f"{px:.5f} nm | amp={py:.3e}")
        
        # Smart Selection Logic
        if target_center is not None and peaks_list and not multi_fit:
            # Cari index yang paling dekat dengan target_center
            best_idx = 0
            min_dist = float('inf')
            for i, (px, py) in enumerate(peaks_list):
                dist = abs(px - target_center)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            # Select ONLY the best candidate
            listw.item(best_idx).setSelected(True)
        else:
            # Fallback: Select All (old behavior) or strongest?
            # User implies they want specific selection. Let's select all if no target known.
            listw.selectAll()
            
        layout.addWidget(listw)

        add_layout = QtWidgets.QHBoxLayout()
        self.manual_edit = QtWidgets.QLineEdit()
        self.manual_edit.setPlaceholderText("λ manual (nm)")
        add_btn = QtWidgets.QPushButton("Tambah")
        # tombol untuk memuat draft pilihan terakhir
        use_prev_btn = QtWidgets.QPushButton("Gunakan draft terakhir")
        use_prev_btn.setEnabled(bool(self.last_selected_peaks))
        clear_prev_btn = QtWidgets.QPushButton("Hapus draft")
        clear_prev_btn.setEnabled(bool(self.last_selected_peaks))
        add_layout.addWidget(self.manual_edit)
        add_layout.addWidget(add_btn)
        add_layout.addWidget(use_prev_btn)
        add_layout.addWidget(clear_prev_btn)
        layout.addLayout(add_layout)

        def add_manual_peak():
            text = self.manual_edit.text().strip()
            if not text:
                return
            # dukung banyak angka dipisah koma/spasi
            raw_tokens = [t for t in text.replace(";", ",").split(",") if t.strip()]
            added_any = False
            for tok in raw_tokens:
                try:
                    lam = float(tok.strip().replace(" ", ""))
                except ValueError:
                    continue
                amp = peaks_list[-1][1] if peaks_list else 1.0
                peaks_list.append((lam, amp))
                listw.addItem(f"{lam:.5f} nm | amp≈{amp:.3e}")
                listw.item(listw.count() - 1).setSelected(True)
                added_any = True
            if not added_any:
                QtWidgets.QMessageBox.warning(self, "Input salah", "λ harus angka (boleh banyak, pisah koma).")
                return
            self.manual_edit.clear()

        add_btn.clicked.connect(add_manual_peak)

        def use_previous():
            if not self.last_selected_peaks:
                return
            for lam, amp in self.last_selected_peaks:
                peaks_list.append((lam, amp))
                listw.addItem(f"{lam:.5f} nm | amp≈{amp:.3e}")
                listw.item(listw.count() - 1).setSelected(True)

        use_prev_btn.clicked.connect(use_previous)
        clear_prev_btn.clicked.connect(lambda: self._clear_last_draft(dialog))

        # -- Live Preview Logic --
        def on_selection_change():
            if not preview_callback: return
            
            # Get current selection
            sel = []
            for idx in range(listw.count()):
                item = listw.item(idx)
                if item.isSelected():
                    sel.append(peaks_list[idx])
            
            # Call preview
            if sel:
                preview_callback(sel)
            else:
                # Clear fit curve if nothing selected
                preview_callback([]) 

        if preview_callback:
            listw.itemSelectionChanged.connect(on_selection_change)

        # -- Buttons --
        # Revert to standard OK/Cancel but keep preview
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)
        
        def on_accept():
            # Save position for next dialog
            self.last_dialog_geometry = dialog.geometry()
            dialog.accept()
            
        btns.accepted.connect(on_accept)
        btns.rejected.connect(dialog.reject)
        
        # Trigger initial preview
        if preview_callback:
            on_selection_change()

        ret = dialog.exec()
        
        if ret != QtWidgets.QDialog.Accepted:
            return [] # Cancel

        # Return final selection
        selected = []
        for idx in range(listw.count()):
            item = listw.item(idx)
            if item.isSelected():
                px, py = peaks_list[idx]
                selected.append((px, py))
        if selected:
            # simpan sebagai draft terakhir untuk dipakai lagi
            self.last_selected_peaks = selected
        return selected

    def _clear_last_draft(self, parent_dialog=None):
        """Hapus draft puncak terakhir dan update tombol dialog jika ada."""
        self.last_selected_peaks = []
        if parent_dialog:
            for btn in parent_dialog.findChildren(QtWidgets.QPushButton):
                if btn.text() in {"Gunakan draft terakhir", "Hapus draft"}:
                    btn.setEnabled(False)

    def fit_voigt_roi(self):
        if models is None or fitting is None:
            QtWidgets.QMessageBox.warning(self, "Astropy missing", "Install astropy to use Voigt fit.")
            return

        # Trigger display update (affects all if syncing logic exists, or just global flag)
        self.use_full_display = True
        self.skip_revert_once = True
        self.apply_display_mode()
        self.render_plot()
        self.statusBar().showMessage("Fit 3 Panel Full-Res...", 3000)
        
        self.last_fit_results = {} # index -> result
        self.last_fit_result = None # legacy compat
        
        combined_report = ""
        
        for idx, panel in enumerate(self.panels):
            if panel.current_file is None:
                continue
                
            res, msg, fit_data = self._fit_single_panel(panel)
            
            head = f"--- Panel {idx+1}: {panel.current_file.name} ---"
            if res:
                self.last_fit_results[idx] = res
                if idx == 0: self.last_fit_result = res
                combined_report += f"{head}\n{msg}\n\n"
                
                # Update panel fit curve
                x_fit, y_fit = fit_data
                panel.fit_curve.setData(x_fit, y_fit)
            else:
                combined_report += f"{head}\nFit Failed: {msg}\n\n"
                panel.fit_curve.setData([], [])
                
        self.result_view.setText(combined_report)
        # Enable save if any result
        if self.last_fit_results:
            self.collect_btn.setEnabled(True)

    def _fit_single_panel(self, panel, silent: bool = False, context_info: str = None):
        """Helper to fit a single panel's data in its current ROI."""
        data_wl_full = panel.data_wl_full
        data_inten_full = panel.data_inten_full
        
        if data_wl_full is None or data_inten_full is None:
            return None, "No Data", None
            
        r0, r1 = panel.region.getRegion()
        # Ensure mask is valid
        if r0 >= r1: return None, "ROI Interval Invalid", None
            
        mask = (data_wl_full >= r0) & (data_wl_full <= r1)
        if not np.any(mask):
            return None, "ROI Empty", None
            
        x = data_wl_full[mask]
        y = data_inten_full[mask]
        
        finite_mask = np.isfinite(x) & np.isfinite(y)
        x = x[finite_mask]
        y = y[finite_mask]
        
        if x.size < 5: return None, "Points < 5", None

        # Interpolasi
        try:
            n_grid = max(200, int((x.max() - x.min()) / 0.001))
            x_dense = np.linspace(x.min(), x.max(), n_grid)
            y_dense = np.interp(x_dense, x, y)
        except:
            return None, "Interpolation Failed", None
        
        # Baseline
        edge_n = max(3, int(0.05 * x.size))
        edges = np.r_[y_dense[:edge_n], y_dense[-edge_n:]]
        baseline = float(np.median(edges[np.isfinite(edges)])) if edges.size > 0 else 0.0
        y_net = y_dense - baseline
        y_net = np.where(np.isfinite(y_net), y_net, 0.0)
        
        # DEBUG trace
        print(f"[DEBUG_FIT] Panel {panel.current_file.name}: Points={x.size}, Baseline={baseline:.3f}, MaxYNet={y_net.max():.3f}")

        if not np.any(y_net > 0): return None, "No Signal", None
        
        # Find peaks (using Viewer's params)
        min_amp, min_dist, rel_prom, max_peaks, _ = self._get_peak_params()
        
        peaks = self._find_peaks(x_dense, y_net, min_amp=min_amp, min_dist=min_dist, rel_prom=rel_prom, max_peaks=max_peaks)
        print(f"[DEBUG_FIT] Panel {panel.current_file.name}: Found {len(peaks)} peaks. (min_amp={min_amp})")

        if not peaks: return None, "No Peaks Found", None
        
        roi_center = (r0 + r1) / 2.0
        selected_peaks = self._choose_peaks(peaks, silent=silent, target_center=roi_center, context_info=context_info)
        if not selected_peaks: return None, "Peaks Rejected", None
        
        def clamp(val, lo, hi): return max(lo, min(hi, val))
        amp_init = clamp(float(y_net.max()), 1e-6, float(y_net.max()) * 2.0)
        span = x_dense.max() - x_dense.min()
        fwhm_init = clamp(span * 0.05, 0.01, 1.0)
        tol_x = 0.02; fwhm_max = 0.08

        def build_model(kind):
            model = None
            for cx, _ in selected_peaks:
                if kind == "voigt":
                    comp = models.Voigt1D(x_0=cx, amplitude_L=amp_init, fwhm_L=fwhm_init, fwhm_G=fwhm_init,
                        bounds={"x_0": (cx-tol_x, cx+tol_x), "amplitude_L": (0, None), "fwhm_L": (0.0005, fwhm_max), "fwhm_G": (0.0005, fwhm_max)})
                else: # gauss
                    comp = models.Gaussian1D(amplitude=amp_init, mean=cx, stddev=fwhm_init/2.355,
                        bounds={"mean": (cx-tol_x, cx+tol_x), "amplitude": (0, None), "stddev": (0.0002, fwhm_max/2.355)})
                model = comp if model is None else model + comp
            return model

        fitter = fitting.LevMarLSQFitter()
        
        # -- Define Preview Callback --
        def run_preview(selection):
            """Callback for live preview inside dialog"""
            if not selection:
                panel.fit_curve.setData([], [])
                QtWidgets.QApplication.processEvents()
                return

            # Build temp model
            temp_model = None
            for cx, _ in selection:
                 # Replicated model build logic (simplified for preview speed? No, use full)
                 # Note: Reuse amp_init, etc from outer scope
                 if True: # voigt
                     comp = models.Voigt1D(x_0=cx, amplitude_L=amp_init, fwhm_L=fwhm_init, fwhm_G=fwhm_init,
                         bounds={"x_0": (cx-tol_x, cx+tol_x), "amplitude_L": (0, None), "fwhm_L": (0.0005, fwhm_max), "fwhm_G": (0.0005, fwhm_max)})
                 temp_model = comp if temp_model is None else temp_model + comp
            
            # Fit
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    fitted_prev = fitter(temp_model, x_dense, y_net)
                    y_prev = fitted_prev(x_dense)
                    panel.fit_curve.setData(x_dense, y_prev)
                    QtWidgets.QApplication.processEvents()
            except: 
                panel.fit_curve.setData([], [])

        if not peaks: return None, "No Peaks Found", None
        
        roi_center = (r0 + r1) / 2.0
        selected_peaks = self._choose_peaks(peaks, silent=silent, target_center=roi_center, 
                                            context_info=context_info,
                                            preview_callback=run_preview,
                                            multi_fit=self.multi_peak_cb.isChecked())
                                            
        if not selected_peaks: return None, "Peaks Rejected", None
        
        # Final Fit (re-run to get full output)
        model = build_model("voigt")
        import warnings
        fitted_local = None; y_fit = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                fitted_local = fitter(build_model("voigt"), x_dense, y_net)
        except: pass
        
        model_used = "Voigt"
        if fitted_local is None:
             try: fitted_local = fitter(build_model("gauss"), x_dense, y_net)
             except: pass
             model_used = "Gaussian"
             
        if fitted_local is None: return None, "Fit Exception", None
        
        y_fit = fitted_local(x_dense)
        area_total = float(np.trapezoid(y_fit, x_dense))
        peak_total = float(y_fit.max())
        
        # find center
        center = None
        for attr in ("x_0", "mean", "x_0_0", "mean_0"):
             if hasattr(fitted_local, attr): center = float(getattr(fitted_local, attr).value); break
        if center is None: center = float(x_dense[np.argmax(y_fit)])
        
        # Calc FWHM numeric
        half = peak_total / 2.0
        above = np.where(y_fit >= half)[0]
        fwhm = float(x_dense[above.max()] - x_dense[above.min()]) if above.size >= 2 else 0.0

        msg = f"{model_used} | C={center:.4f} | A={area_total:.3e} | P={peak_total:.3e} | W={fwhm:.4f}"
        fit_message = fitter.fit_info.get("message", "")
        if fit_message: msg += f" | {fit_message}"
        
        # Components info
        comp_rows = []
        try:
            n_sub = getattr(fitted_local, "n_submodels", 0)
            comps = [fitted_local[i] for i in range(n_sub)] if n_sub > 1 else [fitted_local]
            for idx, comp in enumerate(comps):
                 y_comp = comp(x_dense)
                 area_comp = float(np.trapezoid(y_comp, x_dense))
                 peak_comp = float(y_comp.max())
                 c_val = None
                 for attr in (f"x_0_{idx}", f"mean_{idx}", "x_0", "mean"):
                      if hasattr(comp, attr): c_val = float(getattr(comp, attr).value); break
                 if c_val is None: c_val = float(x_dense[np.argmax(y_comp)])
                 
                 half_c = peak_comp / 2.0
                 above_c = np.where(y_comp >= half_c)[0]
                 fwhm_c = float(x_dense[above_c.max()] - x_dense[above_c.min()]) if above_c.size >= 2 else 0.0
                 
                 comp_rows.append((idx + 1, c_val, area_comp, peak_comp, fwhm_c))
        except: pass

        # Result Dict
        result = {
            "center": center, "area": area_total, "peak": peak_total, "fwhm": fwhm, "roi": (r0, r1),
            "timestamp": datetime.now().isoformat(timespec="seconds"), "file": panel.current_file.name,
            "model": model_used,
            "components": comp_rows, "message": msg
        }
        return result, msg, (x_dense, y_fit)

    def update_sim_labels(self, r0: float, r1: float, y_max: float):
        """Tampilkan label puncak simulasi hanya di ROI."""
        # hapus label lama
        if hasattr(self, "sim_label_items"):
            for item in self.sim_label_items:
                try:
                    self.plot_roi.removeItem(item)
                except Exception:
                    pass
        self.sim_label_items = []
        if not getattr(self, "sim_lines", None):
            return
        if not self.sim_overlay_cb.isChecked():
            return
        if not self.sim_label_cb.isChecked():
            return
        if not hasattr(self, "sim_curve_roi"):
            return
        x_sim, y_sim = self.sim_curve_roi.getData()
        if x_sim is None or y_sim is None or len(x_sim) == 0:
            return
        x_sim = np.array(x_sim)
        y_sim = np.array(y_sim)
        mask_region = (x_sim >= r0) & (x_sim <= r1)
        if not np.any(mask_region):
            return
        for tup in self.sim_lines:
            if len(tup) >= 3:
                wl, _, label = tup[0], tup[1], tup[2]
            elif len(tup) == 2:
                wl, _ = tup[0], tup[1]
                label = "Sim"
            else:
                continue
            if wl < r0 or wl > r1:
                continue
            # ambil amplitude dari kurva simulasi ROI dengan interpolasi
            try:
                y_val = float(np.interp(wl, x_sim, y_sim))
            except Exception:
                y_val = 0.0
            text = f"{label} {wl:.2f} nm"
            ti = pg.TextItem(text, color="orange", anchor=(0.5, 1.0))
            ti.setPos(wl, y_val + 0.03 * y_max)
            self.plot_roi.addItem(ti)
            self.sim_label_items.append(ti)

    def fetch_nist_lines(self):
        """Fetch lines from local HDF5/NIST database."""
        elem = self.nist_elem_edit.text().strip()
        ion_mode = self.nist_ion_combo.currentText()
        min_aki_str = self.nist_min_aki.text().strip()
        
        if not elem:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please enter an element symbol.")
            return

        try:
            min_aki = float(min_aki_str) if min_aki_str else 0.0
        except ValueError:
            min_aki = 0.0
            
        hdf_path = Path(__file__).parent / "nist_lines_all.h5"
        if not hdf_path.exists():
            # Fallback check
            hdf_path = Path("nist_lines_all.h5")
        
        if not hdf_path.exists():
             QtWidgets.QMessageBox.critical(self, "Data Error", f"NIST Database not found at {hdf_path}")
             return
             
        try:
            # Determine species number query
            # Determine species number query
            sp_nums = []
            if ion_mode == "I": sp_nums = [1]
            elif ion_mode == "II": sp_nums = [2]
            elif ion_mode == "III": sp_nums = [3]
            else: sp_nums = [] # All (no filter, or 1,2,3 etc)
            
            # Using pandas to read efficiently
            import pandas as pd
            # Use 'where' clause for efficiency if possible, but element is simple filter
            # store keys usually 'nist_lines'. 
            # Note: The previously inspected 'cfl.py' used 'nist_lines_all.h5'.
            # Let's inspect keys first if unsure, but assuming standard format.
            # If 'element' is a column.
            
            # We'll read the whole table with a where clause if indexed, or filter after.
            # Assuming 'element' column exists.
            
            self.statusBar().showMessage(f"Fetching NIST data for {elem}...", 2000)
            QtWidgets.QApplication.processEvents()
            
            # Try to optimize load
            # Note: PyTables query syntax "element == 'Fe'"
            try:
                df = pd.read_hdf(hdf_path, key='nist_lines', where=f"element == '{elem}'")
            except Exception:
                # Fallback if not indexed or key differs
                 df = pd.read_hdf(hdf_path)
                 df = df[df['element'] == elem]
                 
            if df.empty:
                QtWidgets.QMessageBox.information(self, "No Data", f"No lines found for {elem}")
                return
                
            # Filter species
            if sp_nums:
                df = df[df['sp_num'].isin(sp_nums)]
                
            # Filter Aki
            if 'Aki(s^-1)' in df.columns:
                 df['Aki(s^-1)'] = pd.to_numeric(df['Aki(s^-1)'], errors='coerce').fillna(0)
                 df = df[df['Aki(s^-1)'] >= min_aki]
                 
            # Sort by Aki desc or Wavelength?
            # Usually sorting by Wavelength is better for viewing
            if 'ritz_wl_air(nm)' in df.columns:
                 df['ritz_wl_air(nm)'] = pd.to_numeric(df['ritz_wl_air(nm)'], errors='coerce')
                 df = df.dropna(subset=['ritz_wl_air(nm)'])
                 df = df[(df['ritz_wl_air(nm)'] >= 200) & (df['ritz_wl_air(nm)'] <= 900)]
                 df = df.sort_values('ritz_wl_air(nm)')
                 
            # Convert to list of dicts
            self.target_lines = []
            for _, row in df.iterrows():
                wl = row.get('ritz_wl_air(nm)', float('nan'))
                ek = row.get('Ek(eV)', float('nan'))
                # Calculate Ei (Lower Energy Level): Ek - Energy_photon
                # E_photon (eV) = 1239.84193 / lambda (nm)
                ei = float('nan')
                if np.isfinite(wl) and np.isfinite(ek) and wl > 0:
                     ei = ek - (1239.84193 / wl)
                     if ei < 0: ei = 0.0 # Clip small negs due to rounding
                
                self.target_lines.append({
                    "element": row.get('element', ''),
                    "sp_num": row.get('sp_num', ''),
                    "wavelength": wl,
                    "aki": row.get('Aki(s^-1)', float('nan')),
                    "ek": ek,
                    "gk": row.get('g_k', float('nan')),
                    "ei": ei
                })
                
            self.statusBar().showMessage(f"Fetched {len(self.target_lines)} lines for {elem}.", 3000)
            self.refresh_lines_table()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Fetch Error", str(e))

    def refresh_lines_table(self):
        """Gabungkan NIST targets + koleksi puncak dalam satu tabel."""
        
        # Prepare list of items to display
        rows = []
        
        # Mapping for matching:
        # We want to find if any Collection item matches a Target line.
        # Collection items have 'center'.
        
        tol = 0.10  # nm tolerance for matching
        
        # Simplify: Iterate targets, find best collection match
        # AND: Iterate collections, lists those not matched?
        # User output requirement: "Lines target... jadikan lebih general... masukan data seperti Aki gk Ek"
        # Prioritize showing the TARGET lines.
        
        matched_collections = set()
        
        for tgt in self.target_lines:
            wl = tgt['wavelength']
            best_match = None
            min_d = tol
            
            # Find closest collection peak
            for idx, col in enumerate(self.collection_data):
                c = col.get('center', float('nan'))
                if not np.isfinite(c): continue
                d = abs(c - wl)
                if d < min_d:
                    min_d = d
                    best_match = (col, d)
            
            row_data = {
                "Element": tgt['element'],
                "Ion": tgt.get('ion', str(tgt.get('sp_num', '?'))),
                "Wavelength": wl,
                "Aki": tgt.get('aki', ''),
                "Ei": tgt.get('ei', ''),
                "Ek": tgt.get('ek', ''),
                "File": "",
                "Center": "",
                "Center": "",
                "Peak": "",
                "Area": "",
                "ROI": "",
                "Status": "Target",
                "Delta": ""
            }
            
            if best_match:
                col, d = best_match
                row_data["File"] = col.get("file", "")
                row_data["Center"] = col.get("center", "")
                row_data["Peak"] = col.get("peak", "")
                row_data["Area"] = col.get("area", "")
                r0, r1 = col.get("roi", (0,0))
                row_data["ROI"] = f"{r0:.2f}-{r1:.2f}"
                row_data["Status"] = "Matched"
                row_data["Delta"] = f"{d:.4f}"
                # Mark collection as matched if needed (optional)
            
            rows.append(row_data)
            
        # Add unmatched collection items? (Optional, maybe confusing if mixing different elements)
        # User requested filtering for SPECIFIC element. So non-matched items from other elements shouldn't contaminate list.
        # But if we have collected items for THIS element that are not in NIST target list?
        # Let's stick to showing TARGETS primarily.
        
        # Setup Table
        self.lines_table.setRowCount(len(rows))
        self.lines_table.setColumnCount(13)
        self.lines_table.setHorizontalHeaderLabels(
            ["Element", "Ion", "λ (nm)", "Aki", "Ei (eV)", "Ek (eV)", "File", "Center fit", "Peak", "Area", "ROI", "Status", "Δλ"]
        )
        
        for r, d in enumerate(rows):
            ei_val = d['Ei']
            ek_val = d['Ek']
            str_ei = f"{ei_val:.3f}" if isinstance(ei_val, float) and np.isfinite(ei_val) else ""
            str_ek = f"{ek_val:.3f}" if isinstance(ek_val, float) and np.isfinite(ek_val) else ""
            
            self.lines_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(d['Element'])))
            self.lines_table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(d['Ion'])))
            self.lines_table.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{d['Wavelength']:.3f}"))
            self.lines_table.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{d['Aki']:.2e}"))
            self.lines_table.setItem(r, 4, QtWidgets.QTableWidgetItem(str_ei))
            self.lines_table.setItem(r, 5, QtWidgets.QTableWidgetItem(str_ek))
            self.lines_table.setItem(r, 6, QtWidgets.QTableWidgetItem(str(d['File'])))
            
            c_item = QtWidgets.QTableWidgetItem(f"{d['Center']:.4f}" if d['Center'] != "" else "")
            self.lines_table.setItem(r, 7, c_item)
            
            p_item = QtWidgets.QTableWidgetItem(f"{d['Peak']:.2e}" if d['Peak'] != "" else "")
            self.lines_table.setItem(r, 8, p_item)
            
            a_item = QtWidgets.QTableWidgetItem(f"{d['Area']:.2e}" if d['Area'] != "" else "")
            self.lines_table.setItem(r, 9, a_item)
            
            self.lines_table.setItem(r, 10, QtWidgets.QTableWidgetItem(str(d['ROI'])))
            
            status_item = QtWidgets.QTableWidgetItem(str(d['Status']))
            if d['Status'] == "Matched":
                status_item.setBackground(QtGui.QColor("#daffda")) # Light green
            self.lines_table.setItem(r, 11, status_item)
            
            self.lines_table.setItem(r, 12, QtWidgets.QTableWidgetItem(str(d['Delta'])))

            
        self.lines_table.resizeColumnsToContents()

    def save_target_list(self):
        """Removed functionality."""
        pass


    def on_lines_table_double_click(self, item: QtWidgets.QTableWidgetItem):
        """Set ROI dan marker ke λ target dari tabel."""
        row = item.row()
        if row < 0: return
        
        try:
            # Column 2 is Wavelength
            lam_item = self.lines_table.item(row, 2)
            if not lam_item: return
            lam = float(lam_item.text())
            
            width = 0.4
            r0 = lam - width/2
            r1 = lam + width/2
            
            self.region.setRegion((r0, r1))
            self.update_roi_from_region()
            self.update_peak_markers(lam) # Use marker for target
            
            # Reset fitting display
            self.last_fit_results = {}
            for panel in self.panels:
                panel.fit_curve.setData([], [])
                # Reset title color? No, standard plot.
            self.result_view.clear()
            
            self.statusBar().showMessage(f"Jump to {lam:.3f} nm", 2000)
            
        except Exception:
            pass


    def toggle_sim_overlay(self):
        """Tampilkan/sembunyikan overlay simulasi."""
        if not self.sim_overlay_cb.isChecked():
            if hasattr(self, "sim_curve"):
                self.sim_curve.setData([], [])
            if hasattr(self, "sim_curve_roi"):
                self.sim_curve_roi.setData([], [])
            # hapus label simulasi
            if hasattr(self, "sim_label_items"):
                for item in self.sim_label_items:
                    try:
                        self.plot_roi.removeItem(item)
                    except Exception:
                        pass
                self.sim_label_items = []
            self.statusBar().showMessage("Overlay simulasi disembunyikan", 3000)
        else:
            # redraw jika sudah ada data simulasi
            if hasattr(self, "sim_data") and self.sim_data:
                self.update_roi_from_region()
            self.statusBar().showMessage("Overlay simulasi ditampilkan", 3000)

    def refresh_file_list(self, selected_name: Optional[str] = None):
        if self.current_folder is None:
            return
        files = sorted(self.current_folder.glob("*.asc"), key=lambda p: self.natural_key(p.name))
        self.list_files.clear()
        for f in files:
            self.list_files.addItem(f.name)
        self.file_label.setText(f"Folder: {self.current_folder} ({len(files)} file)")
        if files:
            if selected_name:
                for idx, f in enumerate(files):
                    if f.name == selected_name:
                        self.list_files.setCurrentRow(idx)
                        break
                else:
                    self.list_files.setCurrentRow(0)
            else:
                self.list_files.setCurrentRow(0)
        else:
            self.clear_plot("No data")

    def natural_key(self, name: str) -> List:
        """Natural sort key tanpa regex, supaya S2 < S10."""
        key: List = []
        num = ""
        for ch in name:
            if ch.isdigit():
                num += ch
            else:
                if num:
                    key.append(int(num))
                    num = ""
                key.append(ch.lower())
        if num:
            key.append(int(num))
        return key

    def _load_elements_from_hdf(self, path: str) -> List[str]:
        """Ambil daftar unik elemen dari HDF NIST (kolom 'element')."""
        try:
            import pandas as pd  # lokal
            p = Path(path)
            if not p.exists():
                return []
            df = pd.read_hdf(p, "nist_spectroscopy_data", columns=["element"])
            elems = sorted(df["element"].dropna().unique().tolist())
            return elems
        except Exception:
            return []

    def toggle_export(self):
        self.export_enabled = self.export_btn.isChecked()
        self.export_btn.setText("Ekspor: ON" if self.export_enabled else "Ekspor: OFF")

    def save_state(self, prompt_if_missing: bool = True):
        """Simpan parameter GUI + folder saat ini ke JSON (dan ekspor koleksi ke XLSX)."""
        path: Optional[Path] = self.last_session_path
        if path is None or (prompt_if_missing and path is None):
            default_path = Path(self.current_folder or Path.cwd()) / "plot_state.json"
            path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Simpan sesi viewer", str(default_path), "JSON Files (*.json);;All Files (*)"
            )
            if not path_str:
                return
            path = Path(path_str)
            self.last_session_path = path
        try:
            r0, r1 = self.region.getRegion()
        except Exception:
            r0, r1 = None, None
        selected_item = self.list_files.currentItem()
        state = {
            "version": 1,
            "current_folder": str(self.current_folder) if self.current_folder else "",
            "selected_file": selected_item.text() if selected_item else "",
            "roi": {"start": r0, "end": r1},
            "min_amp": self.min_amp_edit.text(),
            "min_dist": self.min_dist_edit.text(),
            "prom": self.prom_edit.text(),
            "max_peaks": self.max_peaks_edit.text(),
            "start_peak": self.start_peak_edit.text(),
            "lock_width": self.lock_width_cb.isChecked(),
            "move_sensitivity": self.move_sensitivity,
            "export_enabled": self.export_enabled,
            "sim_overlay": self.sim_overlay_cb.isChecked(),
            "sim_labels": self.sim_label_cb.isChecked(),
            "comp": self.comp_edit.text(),
            "temp": self.temp_edit.text(),
            "ne": self.ne_edit.text(),
            "fwhm_line": self.fwhm_line_edit.text(),
            "fwhm_conv": self.fwhm_conv_edit.text(),
            "window_size": (self.width(), self.height()),
            "collection_by_file": self.collection_data_by_file,
            "file_metadata_cache": self.file_metadata_cache,
            "nist_elem": self.nist_elem_edit.text(),
            "nist_ion": self.nist_ion_combo.currentText(),
            "nist_min_aki": self.nist_min_aki.text(),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            # ekspor koleksi ke XLSX berdasar nama yang sama
            if self.collection_data:
                xlsx_path = path.with_suffix(".xlsx")
                self.export_collection_session(auto_path=True, path_override=xlsx_path)
            self.statusBar().showMessage(f"Sesi disimpan ke {path}", 4000)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Gagal simpan", str(exc))

    def load_state(self):
        """Muat parameter GUI + folder dari JSON."""
        default_path = Path(self.current_folder or Path.cwd()) / "plot_state.json"
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load sesi viewer", str(default_path), "JSON Files (*.json);;All Files (*)"
        )
        if not path_str:
            return
        try:
            with open(path_str, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.last_session_path = Path(path_str)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Gagal load", str(exc))
            return
        folder = state.get("current_folder")
        selected_name = state.get("selected_file")
        if folder and Path(folder).is_dir():
            self.current_folder = Path(folder)
            self.refresh_file_list(selected_name=selected_name)
        else:
            self.statusBar().showMessage("Folder tidak valid di sesi; lewati pemuatan folder", 4000)
        # set ROI
        try:
            roi = state.get("roi", {}) or {}
            r0 = float(roi.get("start"))
            r1 = float(roi.get("end"))
            if np.isfinite([r0, r1]).all() and r1 > r0:
                self.region.setRegion((r0, r1))
                self.update_roi_from_region()
        except Exception:
            pass
        # load vars
        self.calibration_offsets = state.get("calibration_offsets", {}) # Load Calibration before loading files
        
        # text/checkbox fields
        self.min_amp_edit.setText(str(state.get("min_amp", "")))
        self.min_dist_edit.setText(str(state.get("min_dist", "")))
        self.prom_edit.setText(str(state.get("prom", state.get("rel_prom", "")))) # Use 'prom' or fallback to 'rel_prom'
        self.max_peaks_edit.setText(str(state.get("max_peaks", "")))
        self.start_peak_edit.setText(str(state.get("start_peak", "")))
        self.lock_width_cb.setChecked(bool(state.get("lock_width", False)))
        self.comp_edit.setText(str(state.get("comp", "")))
        self.temp_edit.setText(str(state.get("temp", "")))
        self.ne_edit.setText(str(state.get("ne", "")))
        self.fwhm_line_edit.setText(str(state.get("fwhm_line", "")))
        self.fwhm_conv_edit.setText(str(state.get("fwhm_conv", "")))
        try:
            self.move_sensitivity = float(state.get("move_sensitivity", self.move_sensitivity))
        except Exception:
            pass
        # export + overlay toggles
        exp_state = bool(state.get("export_enabled", self.export_enabled))
        self.export_btn.blockSignals(True)
        self.export_btn.setChecked(exp_state)
        self.export_btn.blockSignals(False)
        self.toggle_export()
        self.sim_overlay_cb.setChecked(bool(state.get("sim_overlay", self.sim_overlay_cb.isChecked())))
        self.sim_label_cb.setChecked(bool(state.get("sim_labels", self.sim_label_cb.isChecked())))
        win_size = state.get("window_size")
        try:
            if isinstance(win_size, (list, tuple)) and len(win_size) == 2:
                w, h = int(win_size[0]), int(win_size[1])
                if w > 100 and h > 100:
                    self.resize(w, h)
        except Exception:
            pass
        # load koleksi puncak jika ada
        collection_map = state.get("collection_by_file", {})
        # legacy support jika masih bentuk list
        if isinstance(collection_map, list):
            key = self.current_file.name if self.current_file else "__default__"
            collection_map = {key: collection_map} # Convert legacy list to dict
        
        self.collection_data_by_file = {} # Reset RAM session
        if isinstance(collection_map, dict):
            for k, v in collection_map.items():
                parsed_list = []
                if isinstance(v, list):
                    for rec in v:
                        try:
                            r0r1 = rec.get("roi", (float("nan"), float("nan")))
                            r0, r1 = r0r1 if isinstance(r0r1, (list, tuple)) and len(r0r1) >= 2 else (float("nan"), float("nan"))
                            entry = {
                                "timestamp": rec.get("timestamp", ""),
                                "file": rec.get("file", ""),
                                "center": float(rec.get("center", float("nan"))),
                                "peak": float(rec.get("peak", float("nan"))),
                                "area": float(rec.get("area", float("nan"))),
                                "fwhm": float(rec.get("fwhm", float("nan"))),
                                "roi": (float(r0), float(r1)),
                                "model": rec.get("model", ""),
                                "png_path": rec.get("png_path", ""),
                                "txt_path": rec.get("txt_path", ""),
                            }
                            parsed_list.append(entry)
                        except Exception:
                            continue
                self.collection_data_by_file[k] = parsed_list
            current_key = self.current_file.name if self.current_file else next(iter(self.collection_data_by_file), "")
            self.collection_data = [] # Reset RAM session
            
            # Determine path_obj based on available information
            path_obj = None
            if self.current_folder and selected_name:
                path_obj = self.current_folder / selected_name
            elif self.current_file: # Fallback if selected_name wasn't found or folder not set
                path_obj = self.current_file

            if path_obj: # Only proceed if path_obj could be determined
                self.current_file = path_obj
                self.load_file_data(path_obj)
                # self.reload_current_file_state() # REMOVED: Managed by load_file_data internally, preventing recursion

            self.refresh_collection_table()
            self.statusBar().showMessage(f"Sesi dimuat dari {path_str}", 4000)

    def run_sim_overlay(self):
        if DataFetcher is None or MixedSpectrumSimulator is None:
            QtWidgets.QMessageBox.warning(self, "Simulasi tidak tersedia", "Modul sim.py tidak bisa diimpor.")
            return
        try:
            temp = float(self.temp_edit.text())
            ne_val = float(eval(self.ne_edit.text(), {"__builtins__": {}}))
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Input salah", "Temperature/n_e harus angka.")
            return
        comps = []
        import re as _re
        tokens = [_t.strip() for _t in _re.split(r"[;,]", self.comp_edit.text()) if _t.strip()]
        has_percent = False
        for token in tokens:
            parts = token.split()
            elem = parts[0]
            perc_val = None
            for p in parts[1:]:
                try:
                    perc_val = float(p)
                    break
                except Exception:
                    continue
            if perc_val is not None:
                has_percent = True
            comps.append((elem, perc_val if perc_val is not None else 0.0))
        if not comps:
            QtWidgets.QMessageBox.warning(self, "Input salah", "Komposisi kosong/tidak valid.")
            return
        if not has_percent:
            even = 100.0 / len(comps)
            comps = [(elem, even) for elem, _ in comps]
        else:
            total = sum(p for _, p in comps)
            if total == 0:
                even = 100.0 / len(comps)
                comps = [(elem, even) for elem, _ in comps]
            else:
                # normalisasi supaya total 100
                comps = [(elem, (p / total) * 100.0) for elem, p in comps]
        nist_path = "nist_data_hog_augmented.h5"
        atomic_path = "atomic_data1.h5"
        if not Path(nist_path).exists() or not Path(atomic_path).exists():
            QtWidgets.QMessageBox.warning(self, "File tidak ada", "Periksa path HDF.")
            return
        # set FWHM optional
        fwhm_line = None
        fwhm_conv = None
        try:
            if self.fwhm_line_edit.text().strip():
                fwhm_line = float(self.fwhm_line_edit.text().replace(",", "."))
            if self.fwhm_conv_edit.text().strip():
                fwhm_conv = float(self.fwhm_conv_edit.text().replace(",", "."))
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Input salah", "FWHM harus angka (nm).")
            return
        try:
            wavelengths, spectrum, comp_info, prom_lines = self._simulate_overlay(comps, temp, ne_val, nist_path, atomic_path, fwhm_line, fwhm_conv)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Simulasi gagal", str(exc))
            return
        if wavelengths is None or spectrum is None:
            QtWidgets.QMessageBox.warning(self, "Simulasi gagal", "Tidak ada spektrum dihasilkan (max=0?).")
            return
        self.sim_data = (wavelengths, spectrum)
        self.sim_meta = comp_info or {}
        # normalisasi format garis simulasi: (wl, intensitas, label)
        sim_lines_raw = []
        for line in prom_lines or []:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            if len(line) >= 3:
                wl, inten, label = line[0], line[1], line[2]
            else:
                wl, inten = line[0], line[1]
                label = "Sim"
            sim_lines_raw.append((float(wl), float(inten), str(label)))
        # pastikan tiap elemen terwakili: ambil top 10 per label, lalu gabungkan dan batasi 300
        grouped = {}
        for wl, inten, label in sim_lines_raw:
            grouped.setdefault(label, []).append((wl, inten, label))
        sim_lines = []
        for label, items in grouped.items():
            items_sorted = sorted(items, key=lambda t: t[1], reverse=True)
            sim_lines.extend(items_sorted[:10])
        # tambahkan sisa berdasar intensitas global hingga batas
        if len(sim_lines) < 300:
            remaining = sorted(sim_lines_raw, key=lambda t: t[1], reverse=True)
            for tup in remaining:
                if len(sim_lines) >= 300:
                    break
                sim_lines.append(tup)
        self.sim_lines = sim_lines
        # overlay di plot utama
        if not hasattr(self, "sim_curve"):
            self.sim_curve = self.plot_full.plot([], pen=pg.mkPen(color="orange", width=1, style=QtCore.Qt.DashLine))
        if self.sim_overlay_cb.isChecked():
            self.sim_curve.setData(wavelengths, spectrum)
            # pastikan y-range menampung overlay, batas atas 110% dari puncak tertinggi (data/overlay)
            data_max_full = float(np.nanmax(np.abs(self.data_inten))) if self.data_inten is not None else 0.0
            overlay_max = float(np.nanmax(np.abs(spectrum))) if spectrum.size else 0.0
            top = max(data_max_full, overlay_max)
            top = top * 1.1 if top > 0 else 1.0
            bottom = 0.0
            # jika ada nilai negatif pada data, biarkan terlihat
            if self.data_inten is not None:
                bottom = min(float(np.nanmin(self.data_inten)), 0.0)
            self.plot_full.setYRange(bottom, top, padding=0)
        else:
            self.sim_curve.setData([], [])
        # tampilkan info komposisi neutral/ion di panel khusus
        if self.sim_meta:
            frac_lines = [
                f"{k}: {v:.1f}%"
                for k, v in self.sim_meta.items()
                if isinstance(v, (int, float)) and ((" I" in k or " II" in k) or k.endswith(" 1") or k.endswith(" 2"))
            ]
            if not frac_lines:
                frac_lines = [f"{k}: {v:.1f}%" for k, v in self.sim_meta.items() if isinstance(v, (int, float))]
            line_info = []
            if getattr(self, "sim_lines", None):
                line_info.append("Garis disimulasikan (atom sp λ):")
                max_lines = 200  # tampilkan semua sampai 200 baris untuk scroll
                total_lines = len(self.sim_lines)
                for tup in self.sim_lines[:max_lines]:
                    if len(tup) >= 3:
                        wl, intensity, label = tup[0], tup[1], tup[2]
                    elif len(tup) == 2:
                        wl, intensity = tup[0], tup[1]
                        label = "Sim"
                    else:
                        continue
                    line_info.append(f"{label} | {wl:.3f} nm | I≈{intensity:.2e}")
                if total_lines > max_lines:
                    line_info.append(f"... ({total_lines - max_lines} baris disembunyikan)")
            lines = [
                f"T = {temp:.0f} K, n_e = {ne_val:.3e} cm^-3",
                "Fraksi Saha:",
                *frac_lines
            ]
            if "missing" in self.sim_meta:
                lines.extend(["", f"Tidak ada data HDF untuk: {self.sim_meta['missing']}"])
            if line_info:
                lines.extend(["", *line_info])
            # self.sim_view.setPlainText("\n".join(lines))
            self.statusBar().showMessage("Overlay simulasi ditambahkan (utama & ROI)", 4000)
        # update ROI overlay
        self.update_roi_from_region()

    def _simulate_overlay(self, comps, temp, ne_val, nist_path, atomic_path, fwhm_line=None, fwhm_conv=None):
        # load ionization energies
        ionization_energies = {}
        import h5py
        with h5py.File(atomic_path, 'r') as f:
            dset = f['elements']
            columns = dset.attrs['columns']
            data = []
            for item in dset[:]:
                row = [
                    item[0],
                    item[1].decode('utf-8'), item[2].decode('utf-8'),
                    item[3].decode('utf-8'), item[4].decode('utf-8'),
                    item[5], item[6].decode('utf-8')
                ]
                data.append(row)
            df_ion = pd.DataFrame(data, columns=columns)
            for _, row in df_ion.iterrows():
                ionization_energies[row["Sp. Name"]] = float(row["Ionization Energy (eV)"])
        if sim_module is not None:
            sim_module.ionization_energies = ionization_energies

        fetcher = DataFetcher(nist_path)
        # siapkan config lokal untuk sigma
        cfg = {}
        try:
            from sim import SIMULATION_CONFIG as _SC  # type: ignore
            cfg.update(_SC)
        except Exception:
            pass
        if fwhm_line is not None:
            cfg["sigma"] = float(fwhm_line) / 2.355
        if fwhm_conv is not None:
            cfg["convolution_sigma"] = float(fwhm_conv) / 2.355
        simulators = []
        delta_E_max_dict = {}
        unique_elements = {elem for elem, _ in comps}
        missing = []
        for elem in unique_elements:
            for ion in [1, 2]:
                nist_data, delta_E = fetcher.get_nist_data(elem, ion)
                if not nist_data:
                    missing.append(f"{elem} {ion}")
                    continue
                delta_E_max_dict[f"{elem}_{ion}"] = delta_E
                ion_energy = ionization_energies.get(f"{elem} {'I' if ion == 1 else 'II'}", 0.0)
                simulator = SpectrumSimulator(nist_data, elem, ion, temp, ion_energy, config=cfg)
                simulators.append(simulator)
        if not simulators:
            return None, None
        mixed = MixedSpectrumSimulator(simulators, ne_val, delta_E_max_dict, config=cfg)
        wl, spec, comp_final, prom_lines = mixed.generate_spectrum(comps, temp)
        if np.nanmax(np.abs(spec)) == 0:
            return None, None, None, None
        # scale to current data max (if ada)
        if self.data_inten is not None and self.data_inten.size > 0:
            current_max = float(np.nanmax(np.abs(self.data_inten)))
            if current_max > 0:
                spec = spec * (current_max / max(np.nanmax(np.abs(spec)), 1e-9))
        # tambahkan info missing
        comp_final = comp_final or {}
        if missing:
            comp_final["missing"] = ", ".join(sorted(missing))
        return wl, spec, comp_final, prom_lines
    # -------- Keyboard controls --------

    def on_mouse_moved_roi(self, evt):
        if self.plot_roi.scene() is None:
            return
        pos = evt[0]  # QPointF
        if not self.plot_roi.sceneBoundingRect().contains(pos):
            return
        vb = self.plot_roi.getViewBox()
        mouse_point = vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        self.coord_label.setText(f"ROI x: {x:.4f} nm, y: {y:.3e}")

    def keyPressEvent(self, event):
        key = event.key()
        # Move ROI left/right
        if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            if self.data_wl is not None:
                r0, r1 = self.region.getRegion()
                width = r1 - r0
                if width <= 0:
                    return
                direction = -1 if key == QtCore.Qt.Key_Left else 1
                step = width * 0.05 * self.move_sensitivity
                new_r0 = r0 + direction * step
                new_r1 = r1 + direction * step
                # clamp to data range
                data_min = float(self.data_wl.min())
                data_max = float(self.data_wl.max())
                if new_r0 < data_min:
                    new_r1 += (data_min - new_r0)
                    new_r0 = data_min
                if new_r1 > data_max:
                    new_r0 -= (new_r1 - data_max)
                    new_r1 = data_max
                if new_r0 < data_min:
                    new_r0, new_r1 = data_min, data_min + width
                self.region.setRegion((new_r0, new_r1))
                return
        # Adjust sensitivity (up/down)
        if key == QtCore.Qt.Key_Up:
            self.move_sensitivity *= 1.2
            self.statusBar().showMessage(f"Sensitivity: x{self.move_sensitivity:.2f}", 1500)
            return
        if key == QtCore.Qt.Key_Down:
            self.move_sensitivity = max(0.2, self.move_sensitivity / 1.2)
            self.statusBar().showMessage(f"Sensitivity: x{self.move_sensitivity:.2f}", 1500)
            return
        # Cmd+S untuk save sesi cepat
        if event.modifiers() & QtCore.Qt.MetaModifier and key == QtCore.Qt.Key_S:
            self.save_state(prompt_if_missing=(self.last_session_path is None))
            return
        
        # Esc to Quit with Save Prompt
        if key == QtCore.Qt.Key_Escape:
             reply = QtWidgets.QMessageBox.question(
                 self,
                 "Keluar",
                 "Simpan sesi sebelum keluar?",
                 QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
             )
             if reply == QtWidgets.QMessageBox.Yes:
                 self.save_state()
                 self.close()
             elif reply == QtWidgets.QMessageBox.No:
                 self.close()
             return

        # File navigation with W/S
        if key in (QtCore.Qt.Key_W, QtCore.Qt.Key_S):
            row = self.list_files.currentRow()
            if row < 0:
                return
            if key == QtCore.Qt.Key_W and row > 0:
                self.list_files.setCurrentRow(row - 1)
            elif key == QtCore.Qt.Key_S and row < self.list_files.count() - 1:
                self.list_files.setCurrentRow(row + 1)
            return
        # Deteksi puncak (D) dan simpan puncak (P)
        if event.modifiers() & QtCore.Qt.ControlModifier:
            if key == QtCore.Qt.Key_S:
                self.save_peak_record()
                return
            if key == QtCore.Qt.Key_O:
                self.import_collection_session()
                return
            if key == QtCore.Qt.Key_I:
                self.import_peaks_dialog()
                return
            if key == QtCore.Qt.Key_E:
                self.export_collection_session()
                return
        if key == QtCore.Qt.Key_D:
            self.update_peak_detection(auto_center=False)
            return
        if key == QtCore.Qt.Key_P:
            self.save_peak_record()
            return
        super().keyPressEvent(event)

    def downsample(self, wl: np.ndarray, inten: np.ndarray, max_points: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce points for lighter plotting."""
        n = wl.size
        if n <= max_points:
            return wl, inten
        idx = np.linspace(0, n - 1, max_points).astype(int)
        return wl[idx], inten[idx]

    def save_peak_record(self):
        """Simpan hasil fit dari semua panel aktif."""
        if not self.last_fit_results:
             QtWidgets.QMessageBox.information(self, "Info", "Belum ada hasil fit untuk disimpan.")
             return
             
        saved_total = 0
        now_str = datetime.now().isoformat(timespec="seconds")
        
        for idx, res in self.last_fit_results.items():
            if idx >= len(self.panels): continue
            panel = self.panels[idx]
            current_file = panel.current_file
            if not current_file: continue
            
            res_file = res.get("file", "")
            if res_file != current_file.name: continue
            
            # Generate PNG if missing
            png_path = res.get("png_path", "")
            if not png_path:
                 out_dir = (self.current_folder or Path.cwd()) / "fit_exports"
                 out_dir.mkdir(parents=True, exist_ok=True)
                 fname = f"fit_{current_file.stem}_p{idx}_{datetime.now().strftime('%H%M%S_%f')}.png"
                 try:
                     p_path = out_dir / fname
                     panel.plot_roi.grab().save(str(p_path))
                     png_path = str(p_path)
                     res["png_path"] = png_path
                 except Exception: png_path = ""

            entries = []
            comps = res.get("components") or []
            
            def create_entry(center_val, area_val, peak_val, roi_tuple, model_val, fwhm_val=float("nan")):
                elem, ion, aki, gk, ek, nist_wl = "", "", "", "", "", ""
                if np.isfinite(center_val) and self.target_lines:
                    candidates = []
                    for t in self.target_lines:
                        if abs(t['wavelength'] - center_val) <= 0.10:
                            candidates.append((abs(t['wavelength'] - center_val), t))
                    if candidates:
                        match = min(candidates, key=lambda x: x[0])[1]
                        elem = match.get('element', '')
                        # 'ion' key holds string "I", "II" etc. from fetch_nist_lines
                        ion = match.get('ion', str(match.get('sp_num', '')))
                        aki = match.get('aki', '')
                        gk = match.get('gk', '')
                        ek = match.get('ek', '')
                        nist_wl = match.get('wavelength', '')
                
                return {
                    "timestamp": now_str, "file": current_file.name,
                    "center": center_val, "peak": peak_val, "area": area_val, "fwhm": fwhm_val,
                    "roi": roi_tuple, "model": model_val, "png_path": png_path,
                    "txt_path": res.get("txt_path", ""),
                    "element": elem, "ion_or_sp": ion, "aki": aki, "gk": gk, "ek_ev": ek, 
                    "nist_wavelength_nm": nist_wl
                }

            if comps:
                for idx_c, c_val, a_val, p_val, fwhm_c in comps:
                    ent = create_entry(c_val, a_val, p_val, res.get("roi"), res.get("model"), fwhm_c)
                    # Filter: Only add if matched element OR no targets loaded (blind mode)
                    # And ensure area is positive (> 1e-9 to avoid near-zero noise)
                    if (ent["element"] or not self.target_lines) and (ent["area"] > 1e-9):
                        entries.append(ent)
            else:
                ent = create_entry(
                    res.get("center", float("nan")), res.get("area", float("nan")), 
                    res.get("peak", float("nan")), res.get("roi"), 
                    res.get("model", ""), res.get("fwhm", float("nan"))
                )
                if (ent["element"] or not self.target_lines) and (ent["area"] > 1e-9):
                    entries.append(ent)
            
            if not entries: continue
            if idx == 0:
                 # Panel 0: Always belongs to the active session/UI
                 if current_file.name == self.current_file.name:
                     self.collection_data.extend(entries)
                     print(f"[DEBUG] Panel 0 (Active): Added {len(entries)} entries to collection.")
                 else:
                     # Fallback
                     self.collection_data.extend(entries)
                     print(f"[DEBUG] Panel 0 (Fallback): Added {len(entries)} entries.")
            else:
                 # Panel 1 & 2: Background save
                 print(f"[DEBUG] Panel {idx} ({current_file.name}): Saving {len(entries)} entries to bg file.")
                 self._save_background_entries(current_file, entries)

            saved_total += len(entries)
        
        if saved_total > 0:
             self.collection_count = len(self.collection_data)
             self.refresh_collection_table()
             self.export_collection_session(auto_path=True)
             self.statusBar().showMessage(f"Disimpan {saved_total} puncak (Cek Terminal untuk detail).", 4000)

    def _save_background_entries(self, file_path: Path, new_entries: list):
        """Helper: Load existing excel -> append -> save for background files."""
        xls_path = self.get_auto_save_path(file_path)
        existing = []
        if xls_path.exists():
            try:
                import pandas as pd
                if xls_path.suffix.lower() == '.csv':
                    df = pd.read_csv(xls_path)
                else:
                    df = pd.read_excel(xls_path)
                for _, row in df.iterrows():
                     entry = {
                        "timestamp": row.get("timestamp", ""),
                        "file": row.get("file", ""),
                        "center": float(row.get("center_nm", row.get("center", float("nan")))),
                        "peak": float(row.get("peak", float("nan"))),
                        "area": float(row.get("area", float("nan"))),
                        "fwhm": float(row.get("fwhm_nm", row.get("fwhm", float("nan")))),
                        "roi": (float(row.get("roi_start_nm", row.get("roi_start", float("nan")))), 
                                float(row.get("roi_end_nm", row.get("roi_end", float("nan"))))),
                        "model": row.get("model", ""),
                        "png_path": row.get("png_path", ""),
                        "txt_path": row.get("txt_path", ""),
                        "element": row.get("element", ""),
                        "ion_or_sp": row.get("ion_or_sp", ""),
                        "aki": row.get("aki", ""),
                        "gk": row.get("gk", ""),
                        "ek_ev": row.get("ek_ev", ""),
                        "nist_wavelength_nm": row.get("nist_wavelength_nm", ""),
                     }
                     existing.append(entry)
            except: pass
        
        full_data = existing + new_entries
        self.export_collection_session(auto_path=True, path_override=xls_path, data_source=full_data)


    def add_collection_entry(self, entry: dict):
        """Tambahkan entri ke koleksi dan refresh tabel."""
        self.collection_data.append(entry)
        self.collection_count = len(self.collection_data)
        if self.current_file is not None:
            self.collection_data_by_file[self.current_file.name] = self.collection_data
        self.refresh_collection_table()
        self.refresh_lines_table()

    def refresh_collection_table(self):
        self.collection_table.setRowCount(len(self.collection_data))
        for row, rec in enumerate(self.collection_data):
            file_item = QtWidgets.QTableWidgetItem(str(rec.get("file", "")))
            center_item = QtWidgets.QTableWidgetItem(f"{rec.get('center', float('nan')):.5f}")
            peak_item = QtWidgets.QTableWidgetItem(f"{rec.get('peak', float('nan')):.3e}")
            area_item = QtWidgets.QTableWidgetItem(f"{rec.get('area', float('nan')):.3e}")
            fwhm_item = QtWidgets.QTableWidgetItem(f"{rec.get('fwhm', float('nan')):.5f}")
            r0, r1 = rec.get("roi", (float("nan"), float("nan")))
            roi_item = QtWidgets.QTableWidgetItem(f"{r0:.3f}-{r1:.3f}")
            png_item = QtWidgets.QTableWidgetItem(str(rec.get("png_path", "")))
            for it in (file_item, center_item, peak_item, area_item, fwhm_item, roi_item, png_item):
                it.setFlags(it.flags() ^ QtCore.Qt.ItemIsEditable)
            self.collection_table.setItem(row, 0, file_item)
            self.collection_table.setItem(row, 1, center_item)
            self.collection_table.setItem(row, 2, peak_item)
            self.collection_table.setItem(row, 3, area_item)
            self.collection_table.setItem(row, 4, fwhm_item)
            self.collection_table.setItem(row, 5, roi_item)
            self.collection_table.setItem(row, 6, png_item)
        self.collection_table.resizeRowsToContents()
        self.refresh_lines_table()

    def on_collection_double_click(self, item):
        row = item.row()
        # Center is at column 1
        item_center = self.collection_table.item(row, 1)
        if item_center:
            try:
                center_val = float(item_center.text())
                self.navigate_to_wavelength(center_val)
            except ValueError:
                pass

    def remove_peak_from_selection(self):
        """Remove selected rows from collection."""
        rows = sorted(set(index.row() for index in self.collection_table.selectedIndexes()), reverse=True)
        if not rows:
            QtWidgets.QMessageBox.information(self, "Info", "Pilih baris yang ingin dihapus.")
            return

        confirm = QtWidgets.QMessageBox.question(
            self, "Konfirmasi Hapus", 
            f"Hapus {len(rows)} puncak dari koleksi?", 
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm == QtWidgets.QMessageBox.Yes:
            for r in rows:
                if 0 <= r < len(self.collection_data):
                    self.collection_data.pop(r)
            self.refresh_collection_table()
            self.statusBar().showMessage(f"Dihapus {len(rows)} puncak.", 3000)

    def navigate_to_wavelength(self, center_nm: float, width: float = 0.6):
        """Helper to jump ROI to a specific wavelength (for Import Preview)."""
        if not self.panels: return
        r0 = center_nm - (width / 2)
        r1 = center_nm + (width / 2)
        
        # 1. Force Full Display (if not already)
        if not self.use_full_display:
             self.toggle_full_display()
             
        # 2. Apply ROI
        self.panels[0].set_region(r0, r1)
        self.update_roi_from_region()
        
        # 3. Add Visual Marker (Yellow line)
        self.update_peak_markers(center_nm)
        
        # Ensure repaint
        QtWidgets.QApplication.processEvents()

    def clear_collection_session(self):
        """Clear all collected peaks."""
        confirm = QtWidgets.QMessageBox.question(
            self, "Konfirmasi", "Hapus semua data puncak di sesi ini?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm == QtWidgets.QMessageBox.Yes:
            self.collection_data = []
            self.collection_data_by_file = {}
            self.refresh_collection_table()
            self.statusBar().showMessage("Sesi dibersihkan.", 3000)

    def fetch_nist_lines(self):
        """Fetch reference lines from NIST (via sim.DataFetcher) for the Table."""
        if DataFetcher is None:
             QtWidgets.QMessageBox.warning(self, "Error", "Modul DataFetcher (sim.py) tidak tersedia.")
             return
             
        elem = self.nist_elem_edit.text().strip()
        ion_str = self.nist_ion_combo.currentText()
        if ion_str == "All":
            ion = None # Handle specific logic or loop? 
            # DataFetcher assumes integer ion. Let's simplify: Only I, II, III allowed.
            # If All, default to I.
            ion = 1
        else:
            ion = {"I":1, "II":2, "III":3}.get(ion_str, 1)
            
        try:
            min_aki = float(self.nist_min_aki.text())
        except ValueError:
            min_aki = 0.0

        try:
            fetcher = DataFetcher("nist_lines_all.h5")
            # get_nist_data returns list of lists: [wl, aki, ek, gk, gi, ...?]
            # We assume it matches formatting needed for target_lines
            # Let's inspect get_nist_data signature or usage elsewhere if possible.
            # Assuming standard structure.
            raw_data, _ = fetcher.get_nist_data(elem, ion)
            
            # Format: [wavelength, aki, ei, ek, ...]
            # We need to map to target_lines dicts
            
            self.target_lines = []
            count = 0
            for row in raw_data:
                # row indices depend on DataFetcher implementation
                # Usually: 0=wl, 1=aki, 2=fi, 3=Ai, 4=Ei, 5=Ek ??
                # Let's assume standard [wl, aki, ek, gk, ...]
                wl = float(row[0])
                aki = float(row[1])
                
                if aki < min_aki: continue
                
                # We store minimal info for table
                entry = {
                    "element": elem,
                    "ion": ion_str,
                    "wavelength": wl,
                    "aki": aki,
                    "ek": row[2] if len(row)>2 else 0.0, 
                    "ei": row[3] if len(row)>3 else 0.0,
                    "gi": row[4] if len(row)>4 else 0.0,
                    "gk": row[5] if len(row)>5 else 0.0,
                }
                self.target_lines.append(entry)
                count += 1
                
            self.target_lines.sort(key=lambda x: x["wavelength"])
            self.refresh_lines_table()
            self.statusBar().showMessage(f"Fetched {count} NIST lines for {elem} {ion_str}", 4000)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Fetch Error", f"Fetch Failed:\n{e}\n\nSee console log.")

    def import_collection_session(self):
        """Import sesi koleksi dari XLSX/CSV."""
        path_str, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import sesi koleksi", str(Path.cwd()), "CSV/Excel Files (*.csv *.xlsx);;All Files (*)"
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            new_data = []
            for _, row in df.iterrows():
                try:
                    r0 = float(row.get("roi_start_nm", row.get("roi_start", float("nan"))))
                    r1 = float(row.get("roi_end_nm", row.get("roi_end", float("nan"))))
                    entry = {
                        "timestamp": row.get("timestamp", ""),
                        "file": row.get("file", ""),
                        "center": float(row.get("center_nm", row.get("center", float("nan")))),
                        "peak": float(row.get("peak", float("nan"))),
                        "area": float(row.get("area", float("nan"))),
                        "fwhm": float(row.get("fwhm_nm", row.get("fwhm", float("nan")))),
                        "roi": (r0, r1),
                        "model": row.get("model", ""),
                        "png_path": row.get("png_path", ""),
                        "txt_path": row.get("txt_path", ""),
                        "element": row.get("element", ""),
                        "ion_or_sp": row.get("ion_or_sp", ""),
                        "aki": row.get("aki", ""),
                        "gk": row.get("gk", ""),
                        "ek_ev": row.get("ek_ev", ""),
                        "nist_wavelength_nm": row.get("nist_wavelength_nm", ""),
                    }
                    new_data.append(entry)
                except Exception:
                    continue
            if new_data:
                self.collection_data = new_data
                self.collection_count = len(self.collection_data)
                self.refresh_collection_table()
                self.statusBar().showMessage(f"Import {len(new_data)} entri koleksi dari {path.name}", 4000)
            else:
                QtWidgets.QMessageBox.information(self, "Import kosong", "Tidak ada entri valid di file.")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Gagal import", str(exc))

    def get_auto_save_path(self, file_obj: Optional[Path] = None) -> Path:
        """Get standard path: raw/Skala-5/b/plot_state_<Sample>.csv."""
        target = file_obj if file_obj is not None else self.current_file
        if not target:
            return Path.cwd() / "plot_state_unknown.csv"
        
        sample_name = target.stem # e.g. S1_raw
        
        # Hardcoding specific path logic based on user interaction context
        # Try to find '0' directory relative to script or data
        base_dir = Path(__file__).parent / "0"
        if not base_dir.exists():
            base_dir = Path(__file__).parent # Fallback
            
        return base_dir / f"plot_state_{sample_name}.csv"

    def reload_current_file_state(self):
        """Auto-load existing plot_state excel for the current file."""
        if not self.current_file: return
        
        path = self.get_auto_save_path()
        if not path.exists():
            self.collection_data = []
            self.refresh_collection_table()
            return
            
        try:
            import pandas as pd
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            if not df.empty:
                # Muat kembali Temperatur bila ada dari file CSV
                first_row = df.iloc[0]
                if "Te_calc_K" in df.columns and pd.notna(first_row.get("Te_calc_K")):
                    fname = getattr(self.current_file, "name", "unknown")
                    if fname not in self.file_metadata_cache:
                        self.file_metadata_cache[fname] = {}
                    self.file_metadata_cache[fname].update({
                        "Te_calc_K": first_row.get("Te_calc_K"),
                        "Te_err_K": first_row.get("Te_err_K", ""),
                        "Te_R2": first_row.get("Te_R2", ""),
                        "Te_Element": first_row.get("Te_Element", "")
                    })
                    
            new_data = []
            for _, row in df.iterrows():
                try:
                     r0 = float(row.get("roi_start_nm", row.get("roi_start", float("nan"))))
                     r1 = float(row.get("roi_end_nm", row.get("roi_end", float("nan"))))
                     entry = {
                        "timestamp": row.get("timestamp", ""),
                        "file": row.get("file", ""),
                        "center": float(row.get("center_nm", row.get("center", float("nan")))),
                        "peak": float(row.get("peak", float("nan"))),
                        "area": float(row.get("area", float("nan"))),
                        "fwhm": float(row.get("fwhm_nm", row.get("fwhm", float("nan")))),
                        "roi": (r0, r1),
                        "model": row.get("model", ""),
                        "png_path": row.get("png_path", ""),
                        "txt_path": row.get("txt_path", ""),
                        # Load atomic metadata back into memory
                        "element": row.get("element", ""),
                        "ion_or_sp": row.get("ion_or_sp", ""),
                        "aki": row.get("aki", ""),
                        "gk": row.get("gk", ""),
                        "ek_ev": row.get("ek_ev", ""),
                        "nist_wavelength_nm": row.get("nist_wavelength_nm", ""),
                     }
                     new_data.append(entry)

                except: continue
                
            self.collection_data = new_data
            self.collection_count = len(self.collection_data)
            self.refresh_collection_table()
            self.statusBar().showMessage(f"Loaded {len(new_data)} peaks from {path.name}", 3000)
            
        except Exception as e:
            self.statusBar().showMessage(f"Error loading state: {e}", 3000)

    
    def calibrate_shift_dialog(self):
        """Dialog to set calibration shift for the current file."""
        if not self.current_file:
            QtWidgets.QMessageBox.warning(self, "Warning", "Load a file first.")
            return

        fname = self.current_file.name
        current_shift = self.calibration_offsets.get(fname, 0.0)
        
        # Simple Dialog: Ask for Shift Amount directly OR Reference Wavelength
        # Let's do simple shift first for robustness
        val, ok = QtWidgets.QInputDialog.getDouble(
            self, "Kalibrasi Shift", 
            f"Set Shift (nm) untuk '{fname}':\n(Positif = Geser ke Kanan)", 
            current_shift, -10.0, 10.0, 4
        )
        
        if ok:
            self.calibration_offsets[fname] = val
            self.statusBar().showMessage(f"Calibration Set: {val:+.4f} nm. Reloading...", 3000)
            # Reload to apply
            self.load_file_data(self.current_file)

    def calibrate_shift_dialog(self):
        """Dialog to set calibration shift for the current file."""
        if not self.current_file:
            QtWidgets.QMessageBox.warning(self, "Warning", "Load a file first.")
            return
            
        dlg = CalibrationDialog(self)
        if dlg.exec():
            # If user clicked Apply in dialog, offsets might be updated.
            # Reload all panels to apply.
            self.load_selected_file()

    def export_collection_session(self, auto_path: bool = False, path_override: Optional[Path] = None, data_source: list = None):
        """Simpan/Update sesi koleksi ke CSV/XLSX per sampel."""
        target_data = data_source if data_source is not None else self.collection_data
        
        if not target_data:
             # Allow saving empty state to "clear" the file
             pass

        # Determine path
        path = path_override if path_override else self.get_auto_save_path()
        
        # We are saving the FULL 'target_data' to the file, essentially overwriting it 
        # with the current complete state. This works as 'append' because 
        # 'collection_data' was loaded from the file initially and new peaks were appended in-memory.
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            
            rows = []
            
            # Fetch metadata for the current file
            target_fname = getattr(self.current_file, "name", "unknown")
            if target_data and len(target_data) > 0:
                 target_fname = target_data[0].get("file", target_fname)
            meta = self.file_metadata_cache.get(target_fname, {})
            
            for rec in target_data:
                r0, r1 = rec.get("roi", (float("nan"), float("nan")))
                center = rec.get("center", float("nan"))
                
                # Use baked-in metadata if available, else blank (don't re-match blindly against potentially wrong target list)
                elem = rec.get("element", "")
                
                # STRICT FILTER (Updated):
                # 1. Reject if Area is small (noise/glitch). Threshold 1e-5 covers the 1e-6/1e-7 noise seen.
                # 2. Reject if NO element is assigned (checked robustly).
                area_val = rec.get("area", 0.0)
                if area_val < 1e-9: continue
                
                elem_str = str(elem).strip()
                if not elem_str or elem_str.lower() == "unknown": continue

                ion = rec.get("ion_or_sp", "")
                aki = rec.get("aki", "")
                gk = rec.get("gk", "")
                ek = rec.get("ek_ev", "")
                nist_wl = rec.get("nist_wavelength_nm", "")

                rows.append({
                    "timestamp": rec.get("timestamp", datetime.now().isoformat(timespec="seconds")),
                    "file": rec.get("file", ""),
                    "element": elem,
                    "ion_or_sp": ion,
                    "roi_start_nm": r0,
                    "roi_end_nm": r1,
                    "center_nm": center,
                    "peak": rec.get("peak", float("nan")),
                    "area": rec.get("area", float("nan")),
                    "fwhm_nm": rec.get("fwhm", float("nan")),
                    "model": rec.get("model", ""),
                    "aki": aki,
                    "gk": gk,
                    "ek_ev": ek,
                    "nist_wavelength_nm": nist_wl,
                    "png_path": rec.get("png_path", ""),
                    "txt_path": rec.get("txt_path", ""),
                    "Te_calc_K": meta.get("Te_calc_K", ""),
                    "Te_err_K": meta.get("Te_err_K", ""),
                    "Te_R2": meta.get("Te_R2", ""),
                    "Te_Element": meta.get("Te_Element", ""),
                })
            df = pd.DataFrame(rows)
            if path.suffix.lower() == '.csv':
                df.to_csv(path, index=False)
            else:
                df.to_excel(path, index=False)

            
            msg = f"Data disimpan ke {path.name} ({len(rows)} peaks)"
            self.statusBar().showMessage(msg, 4000)
            if not auto_path:
                 QtWidgets.QMessageBox.information(self, "Saved", msg)
                 
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Gagal ekspor", str(exc))


    def clear_collection_session(self):
        """Hapus semua entri koleksi."""
        if self.collection_data:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Hapus koleksi",
                "Hapus semua entri koleksi?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self.collection_data = []
        self.collection_count = 0
        if self.current_file is not None:
            self.collection_data_by_file[self.current_file.name] = self.collection_data
        self.refresh_collection_table()
        self.statusBar().showMessage("Koleksi dikosongkan", 3000)

    def on_collection_double_click(self, item):
        """Set ROI ke entri koleksi yang diklik."""
        row = item.row()
        if row < 0 or row >= len(self.collection_data):
            return
        rec = self.collection_data[row]
        r0, r1 = rec.get("roi", (None, None))
        if r0 is None or r1 is None:
            return
        try:
            self.region.setRegion((float(r0), float(r1)))
            self.update_roi_from_region()
            self.statusBar().showMessage(
                f"ROI diset ke koleksi row {row+1} ({r0:.3f}-{r1:.3f} nm)", 3000
            )
        except Exception:
            pass

    def _ensure_peak_markers(self):
        for panel in self.panels:
            if not hasattr(panel, "peak_marker_full"):
                panel.peak_marker_full = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("g", width=1, style=QtCore.Qt.DotLine))
                panel.plot_full.addItem(panel.peak_marker_full)
            if not hasattr(panel, "peak_marker_roi"):
                panel.peak_marker_roi = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("g", width=2))
                panel.plot_roi.addItem(panel.peak_marker_roi)

    def update_peak_markers(self, center: float):
        self._ensure_peak_markers()
        for panel in self.panels:
            panel.peak_marker_full.setValue(center)
            panel.peak_marker_roi.setValue(center)

    def update_peak_detection(self, auto_center: bool = False):
        """Deteksi puncak global dan aktifkan navigator."""
        self.detected_peaks = []
        self.peak_index = -1
        self.prev_peak_btn.setEnabled(False)
        self.next_peak_btn.setEnabled(False)
        if self.data_wl is None or self.data_inten is None:
            return
        finite = np.isfinite(self.data_wl) & np.isfinite(self.data_inten)
        wl = self.data_wl[finite]
        inten = self.data_inten[finite]
        min_amp, min_dist, rel_prom, max_peaks, start_peak = self._get_peak_params()
        peaks = self._find_peaks(wl, inten, min_amp=min_amp, min_dist=min_dist, rel_prom=rel_prom, max_peaks=max_peaks)
        # urutkan untuk navigasi berdasarkan posisi (λ) agar << >> melompat kiri-kanan
        peaks_sorted = sorted(peaks, key=lambda t: t[0])
        self.detected_peaks = peaks_sorted
        if peaks:
            target_idx = 0
            if start_peak is not None:
                for i, (cx, _) in enumerate(peaks_sorted):
                    if cx >= start_peak:
                        target_idx = i
                        break
                else:
                    target_idx = len(peaks_sorted) - 1
            self.peak_index = target_idx
            self.prev_peak_btn.setEnabled(True)
            self.next_peak_btn.setEnabled(True)
            if auto_center or start_peak is not None:
                self.jump_peak(0, absolute_index=target_idx)
            self.statusBar().showMessage(
                f"{len(peaks)} puncak terdeteksi (mulai idx {self.peak_index + 1})", 3000
            )
        else:
            self.statusBar().showMessage("Tidak ada puncak terdeteksi", 3000)

    def jump_peak(self, step: int, absolute_index: Optional[int] = None):
        """Loncat ke puncak sebelumnya/berikutnya secara presisi."""
        if not self.detected_peaks:
            QtWidgets.QMessageBox.information(self, "Belum ada puncak", "Deteksi puncak dulu.")
            return
        if absolute_index is None:
            base = self.peak_index if self.peak_index >= 0 else 0
            idx = base + step
        else:
            idx = absolute_index
        idx = max(0, min(len(self.detected_peaks) - 1, idx))
        self.peak_index = idx
        center, amp = self.detected_peaks[idx]
        r0, r1 = self.region.getRegion()
        width = max(r1 - r0, 0.2)
        new_r0 = center - width / 2.0
        new_r1 = center + width / 2.0
        data_min = float(self.data_wl.min()) if self.data_wl is not None else new_r0
        data_max = float(self.data_wl.max()) if self.data_wl is not None else new_r1
        if new_r0 < data_min:
            new_r1 += (data_min - new_r0)
            new_r0 = data_min
        if new_r1 > data_max:
            new_r0 -= (new_r1 - data_max)
            new_r1 = data_max
        if new_r1 - new_r0 < 0.1:
            mid = (new_r0 + new_r1) / 2.0
            new_r0, new_r1 = mid - 0.05, mid + 0.05
        self.region.setRegion((new_r0, new_r1))
        self.update_peak_markers(center)
        self.statusBar().showMessage(
            f"Peak {idx + 1}/{len(self.detected_peaks)} @ {center:.5f} nm amp≈{amp:.3e}", 3500
        )

    def _get_peak_params(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[float]]:
        """Parse min amplitude, min jarak, prominence relatif, max peaks, dan start λ."""
        min_amp = None
        min_dist = None
        rel_prom = None
        max_peaks = None
        start_peak = None
        txt_amp = self.min_amp_edit.text().strip()
        txt_dist = self.min_dist_edit.text().strip()
        txt_prom = self.prom_edit.text().strip()
        txt_max = self.max_peaks_edit.text().strip()
        txt_start = self.start_peak_edit.text().strip()
        try:
            if txt_amp:
                min_amp = float(txt_amp)
        except Exception:
            min_amp = None
        try:
            if txt_dist:
                min_dist = float(txt_dist)
        except Exception:
            min_dist = None
        try:
            if txt_prom:
                val = float(txt_prom)
                rel_prom = val / 100.0 if val > 1 else val
        except Exception:
            rel_prom = None
        try:
            if txt_max:
                max_peaks = int(float(txt_max))
                if max_peaks <= 0:
                    max_peaks = None
        except Exception:
            max_peaks = None
        try:
            if txt_start:
                start_peak = float(txt_start)
        except Exception:
            start_peak = None
        return min_amp, min_dist, rel_prom, max_peaks, start_peak



    def try_load_temp_from_csv(self, filename: str):
        """Mencoba load Te_K dari b_ALL_TeNe_summary.csv di folder sibling 'c'"""
        # Cek Cache dulu
        if filename in self.file_metadata_cache:
             meta = self.file_metadata_cache[filename]
             if 'Te_K' in meta:
                  te = meta['Te_K']
                  self.temp_edit.setText(f"{te:.0f}")
                  # self.statusBar().showMessage(f"Cached Temp: {te:.0f} K", 2000)
                  return

        if self.current_folder is None: return
        
        parent = self.current_folder.parent 
        # Coba beberapa path kemungkinan
        candidates = [
             parent / "c" / "b_ALL_TeNe_summary.csv",
             parent / "b_ALL_TeNe_summary.csv"
        ]
        
        csv_path = None
        for p in candidates:
             if p.exists():
                  csv_path = p
                  break
        
        if not csv_path:
            return
            
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            match = None
            
            # Cek kolom 'file' (misal "b/filename.asc")
            if 'file' in df.columns:
                 mask = df['file'].astype(str).str.endswith(filename)
                 if mask.any():
                      match = df[mask].iloc[0]
            
            # Fallback cek 'raw_name' (misal "S3-...")
            if match is None and 'raw_name' in df.columns:
                 name_no_ext = os.path.splitext(filename)[0]
                 mask = df['raw_name'].astype(str) == name_no_ext
                 if mask.any():
                      match = df[mask].iloc[0]
            
            if match is not None:
                 if 'Te_K' in match:
                      te = float(match['Te_K'])
                      self.temp_edit.setText(f"{te:.0f}")
                      self.statusBar().showMessage(f"Auto-loaded Temp: {te:.0f} K", 4000)
                      
                      # Save to cache
                      if filename not in self.file_metadata_cache:
                           self.file_metadata_cache[filename] = {}
                      self.file_metadata_cache[filename]['Te_K'] = te
                      
        except Exception as e:
            print(f"Failed to auto-load Temp: {e}") 



    def remove_peak_from_selection(self):
        """Hapus puncak yang terhubung dengan baris terpilih di lines_table."""
        selected_rows = sorted(set(index.row() for index in self.lines_table.selectedIndexes()), reverse=True)
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "Info", "Pilih baris yang statusnya 'Matched' untuk dihapus.")
            return

        cols_to_remove = []
        for r in selected_rows:
            status_item = self.lines_table.item(r, 11) # Status column index
            if not status_item or status_item.text() != "Matched":
                continue
                
            # Get Identification Data
            c_item = self.lines_table.item(r, 7) # Center fit
            f_item = self.lines_table.item(r, 6) # File
            a_item = self.lines_table.item(r, 9) # Area
            
            if not c_item: continue
            try:
                center_val = float(c_item.text())
            except ValueError:
                continue
                
            target_file = f_item.text() if f_item else ""
            try:
                target_area = float(a_item.text()) if a_item else None
            except ValueError:
                target_area = None

            # Cari di collection_data
            found_idx = -1
            min_err = 1e-4
            area_err = 1e-2 # Tolerance for area float comparison
            
            for idx, col in enumerate(self.collection_data):
                # 1. Match File (if available)
                if target_file and col.get('file', '') != target_file:
                    continue
                    
                # 2. Match Center
                c = col.get('center', float('nan'))
                if abs(c - center_val) > min_err:
                    continue
                    
                # 3. Match Area (Critical for distinguishing duplicates/SA candidates)
                if target_area is not None:
                    div = col.get('area', 0)
                    if div == 0: div = 1e-9
                    # Check absolute or relative diff? Area types vary (float).
                    # Simple absolute match with generous tolerance for display rounding
                    # stored area is exact float, display is formatted.
                    # Let's retry exact match logic or close enough
                    if abs(col.get('area', 0) - target_area) > (target_area * 0.01 + 1.0): 
                        # 1% tolerance + 1.0 abs buffer
                        continue
                
                found_idx = idx
                break
            
            if found_idx != -1:
                cols_to_remove.append(found_idx)
        
        if not cols_to_remove:
            QtWidgets.QMessageBox.information(self, "Info", "Tidak ada puncak 'Matched' yang valid di baris terpilih.")
            return

        cols_to_remove = sorted(list(set(cols_to_remove)), reverse=True)
        
        confirm = QtWidgets.QMessageBox.question(
            self, "Konfirmasi Hapus", 
            f"Hapus {len(cols_to_remove)} puncak dari koleksi?", 
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        for idx in cols_to_remove:
            if 0 <= idx < len(self.collection_data):
                self.collection_data.pop(idx)

        # Simpan perubahan
        self.export_collection_session(auto_path=True)
        self.refresh_lines_table()
        # self.refresh_collection_table() # (invisible)
        self.statusBar().showMessage(f"Berhasil menghapus {len(cols_to_remove)} puncak.", 3000)


    def import_peaks_dialog(self):
        """Dialog untuk import puncak dari sample lain (template)."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Pilih File Template (plot_state...)",
            str(self.current_folder) if self.current_folder else "",
            "CSV/Excel Files (*.csv *.xlsx);;All Files (*)"
        )
        
        if file_path:
            self.import_peaks_from_source(Path(file_path))

    def import_peaks_from_source(self, source_path: Path):
        """Propagate peaks from source state to current active sample."""
        try:
            if source_path.suffix.lower() == '.csv':
                df_src = pd.read_csv(source_path)
            else:
                df_src = pd.read_excel(source_path)
            if df_src.empty:
                QtWidgets.QMessageBox.warning(self, "Empty", "File template kosong.")
                return
                
            required_cols = ['center_nm'] # minimal
            if 'center_nm' not in df_src.columns:
                 QtWidgets.QMessageBox.warning(self, "Format Error", "File template tidak valid (missing 'center_nm').")
                 return
            
            # Step 1: Open Import Manager Dialog
            dlg = ImportManagerDialog(self, df_src, self)
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return # Cancelled
                
            # Step 2: Get filtered list
            targets = dlg.get_selected_rows()
            if not targets:
                QtWidgets.QMessageBox.information(self, "Info", "Tidak ada puncak yang dipilih.")
                return
                
            # Step 3: Processing using QProgressDialog
            # Using ProgressDialog is safer than manual processEvents loop to avoid crashes
            progress = QtWidgets.QProgressDialog("Importing Peaks...", "Cancel", 0, len(targets), self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(0)
            
            count = 0
            
            # Dictionary to accumulate background entries per file path to avoid I/O bottlenecks and spam
            bg_entries_accum = {}
            for p_idx, panel in enumerate(self.panels):
                if p_idx > 0 and panel.current_file is not None:
                     bg_entries_accum[panel.current_file] = []
            
            for i, row in enumerate(targets):
                if progress.wasCanceled():
                    break
                progress.setValue(i)
                
                c_target = float(row.get('center_nm', 0))
                if not np.isfinite(c_target): continue
                
                # Define Window +/- 0.3 nm
                w = 0.3
                r0 = c_target - w
                r1 = c_target + w
                
                elem_info = row.get('element', '?')
                ion_info = row.get('ion_or_sp', '?')
                desc = f"Target: {elem_info} {ion_info} @ {c_target:.3f} nm"
                
                # Iterate across ALL active panels
                for p_idx, panel in enumerate(self.panels):
                    if panel.current_file is None: continue
                    
                    # Update ROI region (Needed for fit context)
                    # We do NOT call processEvents here to avoid Bus Error / paint overhead
                    panel.set_region(r0, r1)
                    
                    # Fit Silent
                    res, msg, fit_data = self._fit_single_panel(panel, silent=True, context_info=desc)
                    
                    if res and res['area'] > 1e-12:
                         new_entry = {
                             "center": res['center'],
                             "area": res['area'],
                             "peak": res['peak'],
                             "fwhm": res['fwhm'],
                             "roi": res['roi'],
                             "timestamp": datetime.now().isoformat(timespec="seconds"),
                             "file": panel.current_file.name,
                             "element": elem_info,
                             "ion_or_sp": ion_info,
                             "aki": row.get('aki', ''),
                             "gk": row.get('gk', ''),
                             "ek_ev": row.get('ek_ev', ''),
                             "nist_wavelength_nm": row.get('nist_wavelength_nm', ''),
                             "diff": 0.0
                         }
                         if new_entry.get('nist_wavelength_nm'):
                              try:
                                  nw = float(new_entry['nist_wavelength_nm'])
                                  new_entry['diff'] = new_entry['center'] - nw
                              except: pass

                         # Main session panel: append to collection_data
                         if p_idx == 0:
                             self.collection_data.append(new_entry)
                         else:
                             # Background panels: accumulate to save later
                             bg_entries_accum[panel.current_file].append(new_entry)
                             
                         count += 1
                    else:
                         if p_idx == 0:
                             print(f"[IMPORT_FAIL] Target {c_target:.3f} ({elem_info} {ion_info}): {msg}")

            progress.setValue(len(targets))
            
            # --- CRITICAL FIX: Save MAIN data to disk ---
            self.export_collection_session(auto_path=True)
            self.refresh_collection_table()
            
            # Save all background panels simultaneously in one go
            for bg_file, entries in bg_entries_accum.items():
                 if entries:
                      self._save_background_entries(bg_file, entries)
            
            skipped = len(targets) - count
            msg = f"Berhasil import {count} puncak."
            if skipped > 0:
                msg += f"\nSkipped/Failed: {skipped} puncak.\n(Cek terminal [IMPORT_FAIL] untuk detail)."
                print(f"[IMPORT_DEBUG] Total: {len(targets)}, Success: {count}, Failed: {skipped}")
            
            self.statusBar().showMessage(msg.replace("\n", " "), 5000)
            QtWidgets.QMessageBox.information(self, "Import Selesai", msg)
            
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Error", f"Gagal import: {e}")


    def show_boltzmann_plot(self):
        """Show Boltzmann plot for current collected lines to check for self-absorption."""
        if not self.collection_data:
            QtWidgets.QMessageBox.information(self, "No Data", "Belum ada koleksi puncak disimpan.")
            return

        target_elem = self.nist_elem_edit.text().strip()
        target_ion_ui = self.nist_ion_combo.currentText() # "I", "II", or "All"
        
        x_vals = []
        y_vals = []
        labels = []
        boltz_data = []
        
        KB_EV = 8.617333262e-5
        
        for rec in self.collection_data:
             elem = rec.get('element', '')
             ion_str = str(rec.get('ion_or_sp', ''))
             
             if not elem and self.target_lines:
                  c = rec.get('center', float('nan'))
                  if np.isfinite(c):
                       min_d = 0.1
                       match = None
                       for t in self.target_lines:
                            d = abs(t['wavelength'] - c)
                            if d < min_d:
                                 min_d = d
                                 match = t
                       if match:
                            elem = match['element']
                            sp = match['sp_num']
                            ion_str = "I" if sp == 1 else ("II" if sp == 2 else str(sp))
                            rec['element'] = elem # save missing metadata
                            rec['ion_or_sp'] = ion_str
                            rec['aki'] = match.get('aki', '')
                            rec['gk'] = match.get('gk', '')
                            rec['ek_ev'] = match.get('ek', '')
                            rec['nist_wavelength_nm'] = match.get('wavelength', c)
            
             elem_str = str(elem) if elem is not None else ""
             if target_elem and elem_str.lower() != target_elem.lower():
                  continue
                  
             # Filter by Ion Stage if specified
             # If UI says "I", we skip "II" etc.
             if target_ion_ui in ["I", "II", "III"]:
                  # Normalize record ion string
                  # rec ion might be 1, "1", "I", etc.
                  r_ion = str(ion_str).strip()
                  if r_ion in ["1", "1.0", "I"]: r_ion_norm = "I"
                  elif r_ion in ["2", "2.0", "II"]: r_ion_norm = "II"
                  elif r_ion in ["3", "3.0", "III"]: r_ion_norm = "III"
                  else: r_ion_norm = r_ion
                  
                  if r_ion_norm != target_ion_ui:
                       continue
             
             try:
                  aki = float(rec['aki'])
                  gk = float(rec['gk'])
                  ek = float(rec['ek_ev'])
                  wl = float(rec.get('nist_wavelength_nm', rec['center']))
                  area = float(rec['area'])
                  
                  if aki <= 0 or gk <= 0 or area <= 0: continue
                  
                  # Boltzmann Y-axis: ln( I * lambda / (g * A) )
                  # Note: Some definitions use ln( I / (g * A * nu) ) approx ln( I * lambda / (g * A) )
                  y = np.log( (area * wl) / (aki * gk) )
                  x = ek
                  
                  x_vals.append(x)
                  y_vals.append(y)
                  labels.append(f"{wl:.2f} nm")
                  
                  boltz_data.append({
                      "Wavelength_Obs_nm": float(rec.get('center', wl)),
                      "Wavelength_NIST_nm": wl,
                      "Aki": aki,
                      "gk": gk,
                      "Ek_eV": ek,
                      "Area (I)": area,
                      "y_ln(I*wl/gA)": y
                  })
                  
             except:
                  continue
                  
        if not x_vals:
             QtWidgets.QMessageBox.information(self, "No Points", f"Tidak ada data valid untuk plot Boltzmann elemen {target_elem}. Pastikan data memiliki parameter Aki/gk/Ek.")
             return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Boltzmann Plot: {target_elem}")
        dlg.resize(900, 550)
        
        main_layout = QtWidgets.QHBoxLayout(dlg)
        
        # Left Side: Plot
        plot_layout = QtWidgets.QVBoxLayout()
        lbl_stats = QtWidgets.QLabel("Menghitung regresi...")
        lbl_stats.setAlignment(QtCore.Qt.AlignCenter)
        lbl_stats.setStyleSheet("font-size: 13px; font-weight: bold; color: #E0E0E0;")
        plot_layout.addWidget(lbl_stats)
        
        win = pg.GraphicsLayoutWidget()
        plot_layout.addWidget(win)
        
        plot = win.addPlot(title=f"Boltzmann: {target_elem}")
        plot.setLabel('bottom', "Energy Upper (Ek) [eV]")
        plot.setLabel('left', "ln ( I * λ / gA )")
        plot.showGrid(x=True, y=True)
        
        scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None), brush=pg.mkBrush(0, 150, 255, 200))
        plot.addItem(scatter)
        line_item = pg.PlotDataItem(pen=pg.mkPen('y', width=2), name="Saha-Boltzmann Fit")
        plot.addItem(line_item)
        
        # List to keep track of TextItems for easy cleanup
        text_items = []
        
        # Right Side: List Widget
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(QtWidgets.QLabel("Seleksi Garis Ekstiasi:"))
        
        list_widget = QtWidgets.QListWidget()
        for i, b in enumerate(boltz_data):
            item = QtWidgets.QListWidgetItem(f"{b['Wavelength_NIST_nm']:.2f} nm")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            item.setData(QtCore.Qt.UserRole, i)
            list_widget.addItem(item)
        
        right_layout.addWidget(list_widget)
        
        right_panel = QtWidgets.QWidget()
        right_panel.setLayout(right_layout)
        right_panel.setMaximumWidth(250)
        
        main_layout.addLayout(plot_layout, stretch=3)
        main_layout.addWidget(right_panel, stretch=1)
        
        # Bottom Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_apply = QtWidgets.QPushButton("Gunakan Suhu")
        btn_apply.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 8px; font-size: 14px;")
        btn_apply.setEnabled(False)
        
        btn_close = QtWidgets.QPushButton("Tutup Plot")
        btn_close.clicked.connect(dlg.accept)
        
        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_close)
        
        plot_layout.addLayout(btn_layout)
        
        current_state = {"te": None, "te_err": None, "r2": None, "filtered_boltz": []}
        
        def update_plot():
            # Remove old text labels
            for ti in text_items:
                plot.removeItem(ti)
            text_items.clear()
            
            active_x, active_y, active_lbls = [], [], []
            filtered_boltz = []
            
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item.checkState() == QtCore.Qt.Checked:
                    idx = item.data(QtCore.Qt.UserRole)
                    b = boltz_data[idx]
                    active_x.append(b['Ek_eV'])
                    active_y.append(b['y_ln(I*wl/gA)'])
                    active_lbls.append(f"{b['Wavelength_NIST_nm']:.2f} nm")
                    filtered_boltz.append(b)
                    
            current_state["filtered_boltz"] = filtered_boltz
            
            if len(active_x) >= 2:
                from scipy.stats import linregress
                res = linregress(active_x, active_y)
                slope, intercept = res.slope, res.intercept
                r2 = res.rvalue**2
                err_slope = res.stderr
                
                if slope < 0:
                    calc_te = -1.0 / (KB_EV * slope)
                    te_err = calc_te * (err_slope / abs(slope)) if slope != 0 else 0
                    title_str = f"Te = {calc_te:.0f} ± {te_err:.0f} K  (R² = {r2:.3f})"
                    msg_info = f"Slope = {slope:.3e} ± {err_slope:.3e}"
                    
                    current_state["te"] = calc_te
                    current_state["te_err"] = te_err
                    current_state["r2"] = r2
                    btn_apply.setEnabled(True)
                    btn_apply.setText(f"Gunakan Suhu {calc_te:.0f} K")
                else:
                    title_str = "Invalid Slope > 0 (Self-Absorption)"
                    msg_info = f"Slope = {slope:.3e}"
                    slope = None
                    current_state["te"] = None
                    btn_apply.setEnabled(False)
                    btn_apply.setText("Gunakan Suhu")
            else:
                title_str = "Need ≥ 2 points"
                msg_info = "Pilih minimal 2 garis untuk regresi"
                slope = None
                current_state["te"] = None
                btn_apply.setEnabled(False)
                btn_apply.setText("Gunakan Suhu")
                
            lbl_stats.setText(msg_info)
            plot.setTitle(f"Boltzmann: {target_elem}  |  {title_str}")
            
            # Update scatter data
            points = [{'pos': (x, y), 'data': lbl} for x, y, lbl in zip(active_x, active_y, active_lbls)]
            scatter.setData(points)
            
            if slope is not None and len(active_x) > 0:
                 x_line = np.array([min(active_x), max(active_x)])
                 y_line = slope * x_line + intercept
                 line_item.setData(x_line, y_line)
            else:
                 line_item.setData([], [])
                 
            for x, y, lbl_text in zip(active_x, active_y, active_lbls):
                 t = pg.TextItem(text=lbl_text, color=(200, 200, 200), anchor=(0.5, -0.5))
                 t.setPos(x, y)
                 plot.addItem(t)
                 text_items.append(t)
                 
        list_widget.itemChanged.connect(update_plot)
        update_plot() # Initial plot run
        
        def apply_te():
            calc_te = current_state["te"]
            if calc_te is None: return
            
            self.temp_edit.setText(f"{calc_te:.0f}")
            
            fname = getattr(self.current_file, "name", "unknown")
            if self.collection_data and len(self.collection_data) > 0:
                fname = self.collection_data[0].get("file", fname)
                
            if fname not in self.file_metadata_cache:
                self.file_metadata_cache[fname] = {}
            self.file_metadata_cache[fname].update({
                "Te_calc_K": round(calc_te, 2),
                "Te_err_K": round(current_state["te_err"], 2),
                "Te_R2": round(current_state["r2"], 4),
                "Te_Element": target_elem
            })
            
            self.export_collection_session(auto_path=True)
            
            msg_extra = ""
            final_boltz = current_state["filtered_boltz"]
            if final_boltz:
                try:
                    import pandas as pd
                    folder = self.current_folder or Path.cwd()
                    base_name = fname.replace(".asc", "").replace(".txt", "").replace(".csv", "")
                    out_name = f"Boltzmann_Lines_{base_name}_{target_elem}.xlsx"
                    out_path = folder / out_name
                    df_boltz = pd.DataFrame(final_boltz)
                    df_boltz.to_excel(out_path, index=False)
                    msg_extra = f" (dan {out_name})"
                except Exception as exc:
                    print(f"Failed exporting boltzmann lines: {exc}")
            
            self.statusBar().showMessage(f"Suhu {calc_te:.0f} K disimpan ke log {fname}{msg_extra}", 6000)
            dlg.accept()
            
        btn_apply.clicked.connect(apply_te)
        
        self.boltzmann_win = dlg
        dlg.show()

    def export_boltzmann_diagnosis(self):
        """Export Boltzmann plots for all species to 'Simulasi/Boltzmann_Report/<SampleName>'."""
        if not self.collection_data:
            QtWidgets.QMessageBox.information(self, "No Data", "Belum ada koleksi puncak disimpan.")
            return

        # 1. Prepare Directory
        sample_name = "Unknown_Sample"
        if hasattr(self, 'current_file') and self.current_file:
             sample_name = self.current_file.stem.replace("plot_state_", "").replace("fit_report_", "")
        
        if sample_name == "Unknown_Sample" and self.collection_data:
             sample_name = "Collected_Data"

        report_dir = Path("Simulasi") / "Boltzmann_Report" / sample_name
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Group Data by (Element, Ion)
        grouped_data = {} # (Elem, Ion) -> list of dict
        
        try:
             te_ref = float(self.temp_edit.text())
        except:
             te_ref = None
        
        KB_EV = 8.617333262e-5

        for rec in self.collection_data:
             elem = rec.get('element', '')
             ion_str = str(rec.get('ion_or_sp', ''))
             
             if not elem and self.target_lines:
                  c = rec.get('center', float('nan'))
                  if np.isfinite(c):
                       min_d = 0.1
                       match = None
                       for t in self.target_lines:
                            d = abs(t['wavelength'] - c)
                            if d < min_d:
                                 min_d = d
                                 match = t
                       if match:
                            elem = match['element']
                            sp = match['sp_num']
                            ion_str = "I" if sp == 1 else ("II" if sp == 2 else str(sp))
                            rec['element'] = elem
                            rec['ion_or_sp'] = ion_str
                            rec['aki'] = match.get('aki', '')
                            rec['gk'] = match.get('gk', '')
                            rec['ek_ev'] = match.get('ek', '')
                            rec['nist_wavelength_nm'] = match.get('wavelength', c)
            
             if not elem: continue
             
             # Key (Elem, Ion)
             # Normalize Ion
             r_ion = str(ion_str).strip()
             if r_ion in ["1", "1.0", "I"]: r_ion_norm = "I"
             elif r_ion in ["2", "2.0", "II"]: r_ion_norm = "II"
             else: r_ion_norm = r_ion

             key = (elem, r_ion_norm)
             if key not in grouped_data: grouped_data[key] = []
             
             try:
                  aki = float(rec['aki'])
                  gk = float(rec['gk'])
                  ek = float(rec['ek_ev'])
                  wl = float(rec.get('nist_wavelength_nm', rec['center']))
                  area = float(rec['area'])
                  
                  if aki <= 0 or gk <= 0 or area <= 0: continue
                  
                  y = np.log( (area * wl) / (aki * gk) )
                  x = ek
                  
                  grouped_data[key].append({
                       'x': x, 'y': y, 'wl': wl, 'ek': ek
                  })
             except:
                  continue
        
        if not grouped_data:
             QtWidgets.QMessageBox.warning(self, "No Valid Data", "Tidak ada data valid (Aki/Ek/gk) untuk diekspor.")
             return

        # 3. Generate Plots
        count = 0
        import pyqtgraph.exporters
        
        for (elem, ion), points in grouped_data.items():
             if len(points) < 3: 
                  # print(f"Skipping {elem} {ion}: only {len(points)} points")
                  continue
             
             x_vals = [p['x'] for p in points]
             y_vals = [p['y'] for p in points]
             
             # Fit Linear
             try:
                  slope_fit, intercept_fit = np.polyfit(x_vals, y_vals, 1)
                  if abs(slope_fit) > 1e-9:
                      te_fit = -1.0 / (slope_fit * KB_EV)
                  else:
                      te_fit = float('nan')
             except:
                  slope_fit = None
                  te_fit = float('nan')
             
             # Create headless plot - NEEDS A WIDGET CONTEXT FOR EXPORTER
             win_temp = pg.GraphicsLayoutWidget()
             win_temp.resize(800, 600)
             win_temp.setBackground('w') # Force white background
             plt = win_temp.addPlot(title=f"Boltzmann: {elem} {ion}")
             
             plt.setLabel('bottom', "E_k [eV]")
             plt.setLabel('left', "ln(Iλ/gA)")
             plt.showGrid(x=True, y=True)
             
             # Scatter
             scatter = pg.ScatterPlotItem(x=x_vals, y=y_vals, size=10, pen=None, brush=pg.mkBrush(255, 0, 0, 180))
             plt.addItem(scatter)
             
             # Add labels
             for p in points:
                  lbl_text = f"{p['wl']:.2f}"
                  # Use 'k' or explicitly black for text color on white background
                  txt = pg.TextItem(lbl_text, color='k', anchor=(0.5, 0)) # Anchor to bottom center of text? No, (0.5, 0) means text is above point? 
                  # standard pyqtgraph anchor: (0,0) top-left. (0.5, 1) means x-center, y-bottom matches pos. So text is above pos.
                  txt.setPos(p['x'], p['y'])
                  plt.addItem(txt)
             
             legend = pg.LegendItem(offset=(30, 30))
             legend.setParentItem(plt)
             
             # 1. Fitted Line
             if slope_fit is not None:
                  x_line = [min(x_vals), max(x_vals)]
                  y_line = [slope_fit * xv + intercept_fit for xv in x_line]
                  curve_fit = pg.PlotCurveItem(x_line, y_line, pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine))
                  plt.addItem(curve_fit)
                  legend.addItem(curve_fit, f"Fit: T={te_fit:.0f}K")
                  
             # 2. Reference Line
             if te_ref and te_ref > 100:
                  slope_ref = -1.0 / (KB_EV * te_ref)
                  mx = np.mean(x_vals)
                  my = np.mean(y_vals)
                  int_ref = my - slope_ref * mx
                  
                  x_line = [min(x_vals), max(x_vals)]
                  y_line = [slope_ref * xv + int_ref for xv in x_line]
                  
                  curve_ref = pg.PlotCurveItem(x_line, y_line, pen=pg.mkPen('c', width=2))
                  plt.addItem(curve_ref)
                  legend.addItem(curve_ref, f"Ref: T={te_ref:.0f}K")
             
             # Render
             # Force update to ensure items are placed
             QtWidgets.QApplication.processEvents()
             
             exporter = pyqtgraph.exporters.ImageExporter(plt)
             exporter.parameters()['width'] = 800
             
             safe_ion = str(ion).replace(" ", "_").replace("/", "-")
             out_name = report_dir / f"{elem}_{safe_ion}_boltzmann.png"
             
             try:
                  exporter.export(str(out_name))
                  count += 1
             except Exception as e:
                  print(f"Error exporting {out_name}: {e}")
                  
        if count > 0:
             QtWidgets.QMessageBox.information(self, "Export Sukses", f"Berhasil mengekspor {count} grafik ke:\n{report_dir}\n\nSilakan cek folder tersebut untuk analisis Self-Absorption.")
        else:
             QtWidgets.QMessageBox.warning(self, "Export Gagal", "Tidak ada grafik yang berhasil dibuat. Mungkin poin data per spesies < 3.")


        
    def open_cfl_analyzer(self):
        """Membuka jendela CFL Analyzer dari cfl_gui.py."""
        try:
            from cfl_gui import MainWindow as CFLWindow
            if not hasattr(self, 'cfl_window') or self.cfl_window is None:
                self.cfl_window = CFLWindow()
            self.cfl_window.show()
            self.cfl_window.raise_()
            self.cfl_window.activateWindow()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error CFL", f"Gagal memuat modul CFL Analyzer:\n{e}\nPastikan cfl_gui.py dan library-nya tersedia.")

    def batch_automate_iterations(self):
        from datetime import datetime

        target_dir = Path.cwd() / "0-b"
        target_dir.mkdir(exist_ok=True)
        
        source_dir = Path.cwd() / "0"
        if not source_dir.exists():
            QtWidgets.QMessageBox.warning(self, "Error", "Folder 0 tidak ditemukan.")
            return
            
        asc_files = sorted(list(source_dir.glob("*.asc")))
        if not asc_files:
            QtWidgets.QMessageBox.warning(self, "Error", "Tidak ada file .asc di folder 0.")
            return

        dlg = BatchCalibrationDialog(self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            target_calib_wl = dlg.selected_wl
        else:
            return  # Cancelled batch
        self.multi_peak_cb.setChecked(True)
        
        progress = QtWidgets.QProgressDialog("Memproses Batch (3 iterasi/sampel)...", "Batal", 0, len(asc_files), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        for i_file, asc_path in enumerate(asc_files):
            if progress.wasCanceled(): break
            progress.setValue(i_file)
            
            sample_base = asc_path.stem
            
            import re
            m = re.match(r'(S\d+)', sample_base)
            if not m:
                print(f"[BATCH SKIP] Format tidak valid: {sample_base}")
                continue
            s_id = m.group(1)
            
            # Scan matching base in b/ target
            candidates = list((Path.cwd() / "b").glob(f"plot_state_{s_id}-*b.csv")) + list((Path.cwd() / "b").glob(f"plot_state_{s_id}-*b.xlsx"))
            if not candidates:
                print(f"[BATCH SKIP] {sample_base} - Base state tidak ditemukan di 'b/' untuk {s_id}")
                continue
                
            current_source = candidates[0]
            
            self.load_file_data(asc_path)
            QtWidgets.QApplication.processEvents()
            
            # --- AUTO CALIBRATION ---
            if target_calib_wl is not None:
                p = self.panels[0]
                if p.current_file:
                    fname = p.current_file.name
                    self.calibration_offsets[fname] = 0.0 # reset old
                    x = p.data_wl_full
                    y = p.data_inten_full
                    if x is not None and y is not None:
                        mask = (x >= target_calib_wl - 1.0) & (x <= target_calib_wl + 1.0)
                        x_roi = x[mask]
                        y_roi = y[mask]
                        if len(x_roi) >= 3:
                            idx_max = np.argmax(y_roi)
                            peak_obs = x_roi[idx_max]
                            shift = target_calib_wl - peak_obs
                            self.calibration_offsets[fname] = shift
                            print(f"[BATCH CALIB] {fname} shifted by {shift:+.4f} nm")
                        else:
                            print(f"[BATCH CALIB WARNING] Puncak tidak ditemukan di dekat {target_calib_wl} pada {fname}")
            # ------------------------
            
            progress.setLabelText(f"File {i_file+1}/{len(asc_files)}: {sample_base}")
            QtWidgets.QApplication.processEvents()
            
            try:
                if current_source.suffix.lower() == '.csv':
                    df_src = pd.read_csv(current_source)
                else:
                    df_src = pd.read_excel(current_source)
            except Exception as e:
                print(f"[BATCH] Gagal baca {current_source}: {e}")
                continue
                
            if 'center_nm' not in df_src.columns:
                continue
            
            new_state = []
            for _, row in df_src.iterrows():
                c_target = float(row.get('center_nm', 0))
                if not np.isfinite(c_target): continue
                
                w = 0.3
                r0 = c_target - w
                r1 = c_target + w
                
                desc = f"Target: {row.get('element','?')} {row.get('ion_or_sp','?')} @ {c_target:.3f} nm"
                
                self.panels[0].set_region(r0, r1)
                QtWidgets.QApplication.processEvents()
                
                res, msg, _ = self._fit_single_panel(self.panels[0], silent=True, context_info=desc)
                if res and res['area'] > 1e-12:
                     new_entry = {
                         "center": res['center'],
                         "area": res['area'],
                         "peak": res['peak'],
                         "fwhm": res['fwhm'],
                         "roi": res['roi'],
                         "timestamp": datetime.now().isoformat(timespec="seconds"),
                         "file": self.panels[0].current_file.name,
                         "element": row.get('element', '?'),
                         "ion_or_sp": row.get('ion_or_sp', '?'),
                         "aki": row.get('aki', ''),
                         "gk": row.get('gk', ''),
                         "ek_ev": row.get('ek_ev', ''),
                         "nist_wavelength_nm": row.get('nist_wavelength_nm', ''),
                         "diff": 0.0
                     }
                     if new_entry.get('nist_wavelength_nm'):
                          try:
                              nw = float(new_entry['nist_wavelength_nm'])
                              new_entry['diff'] = new_entry['center'] - nw
                          except: pass
                     new_state.append(new_entry)
            
            iter_output_path = target_dir / f"plot_state_{sample_base}.csv"
            if new_state:
                pd.DataFrame(new_state).to_csv(iter_output_path, index=False)
            else:
                print(f"[BATCH WARNING] {sample_base} gagal mendeteksi puncak apa pun.")
                
            if 'new_state' in locals():
                key = self.current_file.name if self.current_file else sample_base
                self.collection_data_by_file[key] = new_state
                self.collection_data = new_state
                self.refresh_collection_table()
            
        progress.setValue(len(asc_files))
        QtWidgets.QMessageBox.information(self, "Selesai", "Batch Automate selesai diproses dan disimpan ke folder 0-b/.")

class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, parent: PlotViewer):
        super().__init__(parent)
        self.pv = parent
        self.setWindowTitle("Kalibrasi Spektrum")
        self.resize(500, 400)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        
        # 0. Sample Info
        lbl_info = QtWidgets.QLabel("<b>Active Panels:</b>")
        lbl_info.setStyleSheet("font-size: 14px; color: #333; margin-bottom: 5px;")
        self.layout.addWidget(lbl_info)
        
        # 1. Manual Shift
        gb_manual = QtWidgets.QGroupBox("Manual Shift")
        ly_manual = QtWidgets.QVBoxLayout(gb_manual)
        
        self.manual_spinboxes = {}
        for idx, panel in enumerate(self.pv.panels):
            if panel.current_file:
                fname = panel.current_file.name
                curr = self.pv.calibration_offsets.get(fname, 0.0)
                
                row = QtWidgets.QHBoxLayout()
                lbl = QtWidgets.QLabel(f"Panel {idx+1} ({fname}):")
                # Elide long names if needed, but horizontal layout should stretch
                row.addWidget(lbl)
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(-50.0, 50.0)
                spin.setDecimals(4)
                spin.setSingleStep(0.01)
                spin.setValue(curr)
                row.addWidget(spin)
                ly_manual.addLayout(row)
                
                self.manual_spinboxes[fname] = spin
        
        btn_apply_manual = QtWidgets.QPushButton("Apply Manual Shifts")
        btn_apply_manual.clicked.connect(self.apply_manual)
        ly_manual.addWidget(btn_apply_manual)
        self.layout.addWidget(gb_manual)
        
        # 2. Reference Based
        gb_ref = QtWidgets.QGroupBox("Auto-Calibrate by Reference Line")
        ly_ref = QtWidgets.QVBoxLayout(gb_ref)
        
        # Row 1: Element Selector
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(QtWidgets.QLabel("Element:"))
        self.txt_el = QtWidgets.QLineEdit(self.pv.last_calib_element)
        r1.addWidget(self.txt_el)
        r1.addWidget(QtWidgets.QLabel("Ion:"))
        self.combo_ion = QtWidgets.QComboBox()
        self.combo_ion.addItems(["I", "II"])
        self.combo_ion.setCurrentText(self.pv.last_calib_ion)
        r1.addWidget(self.combo_ion)
        
        r1.addWidget(QtWidgets.QLabel("Min Aki:"))
        self.txt_aki = QtWidgets.QLineEdit(self.pv.last_calib_aki)
        self.txt_aki.setFixedWidth(60)
        r1.addWidget(self.txt_aki)
        
        btn_fetch = QtWidgets.QPushButton("Fetch NIST")
        btn_fetch.clicked.connect(self.fetch_lines)
        r1.addWidget(btn_fetch)
        ly_ref.addLayout(r1)
        
        # List of Lines
        self.list_lines = QtWidgets.QListWidget()
        self.list_lines.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        ly_ref.addWidget(self.list_lines)
        
        self.lbl_status = QtWidgets.QLabel("Select element and fetch lines...")
        ly_ref.addWidget(self.lbl_status)
        
        btn_detect = QtWidgets.QPushButton("Detect Peak & Apply Shift")
        btn_detect.clicked.connect(self.run_auto_cal)
        ly_ref.addWidget(btn_detect)
        
        self.layout.addWidget(gb_ref)
        self.setLayout(self.layout)

    def apply_manual(self):
        for fname, spin in self.manual_spinboxes.items():
            self.pv.calibration_offsets[fname] = spin.value()
        self.pv.statusBar().showMessage("Calibration manual shifts applied successfully", 3000)
        self.accept()

    def fetch_lines(self):
        el = self.txt_el.text().strip()
        ion_str = self.combo_ion.currentText()
        aki_str = self.txt_aki.text().strip()
        if not el: return
        
        try:
            min_aki = float(aki_str)
        except ValueError:
            min_aki = 0.0
            
        # Save state to parent window for persistence
        self.pv.last_calib_element = el
        self.pv.last_calib_ion = ion_str
        self.pv.last_calib_aki = aki_str
        
        try:
            # Check availability
            if DataFetcher is None:
                 self.lbl_status.setText("DataFetcher/sim.py not available.")
                 return
                 
            fetcher = DataFetcher("nist_lines_all.h5")
            ion = 1 if ion_str == "I" else 2
            
            data, _ = fetcher.get_nist_data(el, ion)
            self.list_lines.clear()
            
            if not data:
                self.lbl_status.setText(f"No NIST lines found for {el} {ion_str}")
                return
            
            # data is list of tuples/rows?
            # get_nist_data implementation: returns (list_of_lines, delta_E)
            # line: [wl, aki, ek, g_k, g_i ...]
            
            # Sort by Aki/Intensity if possible. 
            # Usually users know the WL. Sort by WL.
            data.sort(key=lambda x: x[0])
            
            filtered_count = 0
            for row in data:
                wl = row[0]
                aki = row[1]
                if aki < min_aki:
                    continue
                item = QtWidgets.QListWidgetItem(f"{wl:.4f} nm  (Aki: {aki:.2e})")
                item.setData(QtCore.Qt.UserRole, wl)
                self.list_lines.addItem(item)
                filtered_count += 1
                
            self.lbl_status.setText(f"Loaded {filtered_count} lines (min Aki {min_aki:.1e}).")
            
        except Exception as e:
             import traceback
             traceback.print_exc()
             QtWidgets.QMessageBox.critical(self, "Error", f"Fetch Failed:\n{e}\n\nSee console for details.")

    def run_auto_cal(self):
        item = self.list_lines.currentItem()
        if not item: return
        ref_wl = item.data(QtCore.Qt.UserRole)
        self.pv.last_calib_ref_wl = ref_wl
        
        panels = [p for p in self.pv.panels if p.current_file is not None]
        if not panels: return
        
        shifts = []
        msg_str = f"Reference Wavelength: {ref_wl:.3f} nm\n\nDetected Shifts per Panel:\n"
        
        for idx, p in enumerate(panels):
            x = p.data_wl_full
            y = p.data_inten_full
            if x is None or y is None: continue
            
            # Undo current shift to seek on raw data
            fname = p.current_file.name
            current_shift = self.pv.calibration_offsets.get(fname, 0.0)
            x_raw = x - current_shift
            
            # Scan +/- 1.0 nm around Ref using raw X
            mask = (x_raw >= ref_wl - 1.0) & (x_raw <= ref_wl + 1.0)
            x_roi = x_raw[mask]
            y_roi = y[mask]
            
            if len(x_roi) < 3:
                msg_str += f"- Panel {idx+1} ({fname}): N/A (no peak found)\n"
                continue
                 
            idx_max = np.argmax(y_roi)
            peak_obs_raw = x_roi[idx_max] # Peak in raw unshifted coords
            
            # Sub-pixel precise fit using mathematical modeling
            try:
                if models is not None and fitting is not None:
                    amp_guess = y_roi[idx_max]
                    base_guess = np.median(y_roi)
                    std_guess = 0.05
                    
                    g_init = models.Gaussian1D(amplitude=amp_guess - base_guess, mean=peak_obs_raw, stddev=std_guess)
                    b_init = models.Const1D(amplitude=base_guess)
                    model_init = g_init + b_init
                    
                    fitter = fitting.LevMarLSQFitter()
                    fitted_model = fitter(model_init, x_roi, y_roi)
                    
                    # Extract sub-pixel mean from the Gaussian component
                    peak_obs_raw = fitted_model.mean_0.value
            except Exception as e:
                print(f"Sub-pixel calibration failed for {fname}, using discrete pixel: {e}")
            
            # Calibration Shift = Reference - Observed_Raw
            new_shift = ref_wl - peak_obs_raw
            shifts.append((fname, new_shift))
            msg_str += f"- Panel {idx+1} ({fname}): shift {new_shift:+.4f} nm\n"
            
        if not shifts:
             self.lbl_status.setText("Failed finding peaks in raw data.")
             return
             
        msg_str += f"\nApply {len(shifts)} shifts to all panels simultaneously?"
        res = QtWidgets.QMessageBox.question(self, "Apply All Auto-Calibrations?", msg_str)
        if res == QtWidgets.QMessageBox.Yes:
            for fname, shift in shifts:
                 self.pv.calibration_offsets[fname] = shift
            self.pv.statusBar().showMessage(f"Auto-Calibration applied to {len(shifts)} panels.", 4000)
            self.accept()

class BatchCalibrationDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Auto-Calibration Target")
        self.resize(400, 450)
        self.selected_wl = None
        self.pv = parent
        
        ly = QtWidgets.QVBoxLayout(self)
        lbl = QtWidgets.QLabel("Pilih garis referensi NIST untuk menggeser posisi X (kalibrasi) seluruh sampel secara otomatis pada background processing.\n\nKlik 'Bypass (Tanpa Offset)' jika tidak mengharapkan kalibrasi data.")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        ly.addWidget(lbl)
        
        # Row 1: Element Selector
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(QtWidgets.QLabel("Element:"))
        self.txt_el = QtWidgets.QLineEdit(getattr(self.pv, 'last_calib_element', ''))
        r1.addWidget(self.txt_el)
        r1.addWidget(QtWidgets.QLabel("Ion:"))
        self.combo_ion = QtWidgets.QComboBox()
        self.combo_ion.addItems(["I", "II", "III"])
        self.combo_ion.setCurrentText(getattr(self.pv, 'last_calib_ion', 'I'))
        r1.addWidget(self.combo_ion)
        
        btn_fetch = QtWidgets.QPushButton("Fetch NIST")
        btn_fetch.clicked.connect(self.fetch_lines)
        r1.addWidget(btn_fetch)
        ly.addLayout(r1)
        
        # List of Lines
        self.list_lines = QtWidgets.QListWidget()
        self.list_lines.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        ly.addWidget(self.list_lines)
        
        self.lbl_status = QtWidgets.QLabel("Select element and fetch lines...")
        ly.addWidget(self.lbl_status)
        
        btn_layout = QtWidgets.QHBoxLayout()
        btn_apply = QtWidgets.QPushButton("Gunakan Garis Ini")
        btn_apply.setStyleSheet("background-color: #0288D1; color: white; font-weight: bold; padding: 5px;")
        btn_apply.clicked.connect(self.apply_wl)
        
        btn_bypass = QtWidgets.QPushButton("Bypass (Tanpa Offset)")
        btn_bypass.clicked.connect(self.bypass)
        
        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_bypass)
        ly.addLayout(btn_layout)
        
    def fetch_lines(self):
        el = self.txt_el.text().strip()
        ion_str = self.combo_ion.currentText()
        if not el: return
        if self.pv:
            self.pv.last_calib_element = el
            self.pv.last_calib_ion = ion_str
            
        try:
            from sim import DataFetcher
            if DataFetcher is None:
                 self.lbl_status.setText("DataFetcher/sim.py not available.")
                 return
                 
            fetcher = DataFetcher("nist_lines_all.h5")
            ion = 1 if ion_str == "I" else (2 if ion_str == "II" else 3)
            data, _ = fetcher.get_nist_data(el, ion)
            self.list_lines.clear()
            
            if not data:
                self.lbl_status.setText(f"No NIST lines found for {el} {ion_str}")
                return
            
            data.sort(key=lambda x: x[0])
            for row in data:
                wl = row[0]
                aki = row[1]
                item = QtWidgets.QListWidgetItem(f"{wl:.4f} nm  (Aki: {aki:.2e})")
                item.setData(QtCore.Qt.UserRole, wl)
                self.list_lines.addItem(item)
                
            self.lbl_status.setText(f"Loaded {len(data)} lines.")
        except Exception as e:
             QtWidgets.QMessageBox.critical(self, "Error", f"Fetch Failed:\n{e}")

    def apply_wl(self):
        item = self.list_lines.currentItem()
        if not item:
            QtWidgets.QMessageBox.warning(self, "Warning", "Pilih garis terlebih dahulu.")
            return
        self.selected_wl = item.data(QtCore.Qt.UserRole)
        self.accept()
        
    def bypass(self):
        self.selected_wl = None
        self.accept()



def main():
    print("=== PLOT.PY VERSION: ERROR HANDLING UPDATED ===")
    app = QtWidgets.QApplication(sys.argv)

    viewer = PlotViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
