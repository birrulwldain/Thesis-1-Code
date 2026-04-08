import os
import sys
from typing import List, Tuple

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from src.libs_physics import (
    DataFetcher,
    LIBSSimulator,
    PhysicsCalculator,
    PlasmaZoneParams,
    SIMULATION_CONFIG,
    TwoZonePlasma,
)


BASE_ELEMENTS = [
    "Al", "Ca", "Fe", "Si", "Mg", "Ti", "Cr", "Mn", "Ni", "Cu", "Li",
    "Na", "K", "O", "H", "N", "Sr", "C", "Rb", "Co", "Pb",
]


class SpectrumCanvas(QWidget):
    MAX_DISPLAY_POINTS = 4000

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.message_label = QLabel()
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setStyleSheet(
            "font-size: 16px; font-weight: 600; color: #334155; background: white;"
        )

        self.plot_full = pg.PlotWidget()
        self.plot_full.setBackground("w")
        self.plot_full.showGrid(x=True, y=True, alpha=0.25)
        self.plot_full.setLabel("bottom", "Wavelength", units="nm")
        self.plot_full.setLabel("left", "Intensity")
        self.plot_full.setMenuEnabled(False)
        self.curve_full = self.plot_full.plot([], [], pen=pg.mkPen(color="#2563eb", width=1.5))

        wl_min, wl_max = SIMULATION_CONFIG["wl_range_nm"]
        self.region = pg.LinearRegionItem(values=(wl_min, min(wl_min + 100.0, wl_max)), movable=True)
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self._update_roi_plot)
        self.plot_full.addItem(self.region)

        self.coord_label = QLabel("ROI: -")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.coord_label.setStyleSheet("color: #475569; font-size: 12px;")

        self.plot_roi = pg.PlotWidget()
        self.plot_roi.setBackground("w")
        self.plot_roi.showGrid(x=True, y=True, alpha=0.25)
        self.plot_roi.setLabel("bottom", "Wavelength (ROI)", units="nm")
        self.plot_roi.setLabel("left", "Intensity")
        self.plot_roi.setMenuEnabled(False)
        self.plot_roi.setMouseEnabled(x=False, y=False)
        self.curve_roi = self.plot_roi.plot([], [], pen=pg.mkPen(color="#dc2626", width=1.7))

        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.summary_label.setStyleSheet(
            "padding: 8px; background: #f8fafc; border: 1px solid #e2e8f0; color: #334155;"
        )

        self.plot_container = QWidget()
        plot_layout = QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.plot_full, 1)
        plot_layout.addWidget(self.coord_label)
        plot_layout.addWidget(self.plot_roi, 1)
        plot_layout.addWidget(self.summary_label)
        self.plot_container.hide()

        layout.addWidget(self.message_label, 1)
        layout.addWidget(self.plot_container, 1)

        self.data_wl_full: np.ndarray | None = None
        self.data_inten_full: np.ndarray | None = None
        self._roi_markers: List[object] = []
        self._roi_labels: List[object] = []
        self._last_metadata: dict | None = None
        self._last_show_labels = True
        self.show_placeholder()

    @staticmethod
    def _format_summary(metadata: dict) -> str:
        return (
            f"Basic: Core T_e={metadata['core_T_e_K']:.0f} K, n_e={metadata['core_n_e_cm3']:.1e} cm^-3, "
            f"RTE={'On' if metadata.get('use_rte', True) else 'Off'}\n"
            f"Advanced: Core T_i={metadata.get('core_T_i_K', metadata['core_T_e_K']):.0f} K, "
            f"Shell T_e={metadata['shell_T_e_K']:.0f} K, Shell T_i={metadata.get('shell_T_i_K', metadata['shell_T_e_K']):.0f} K, "
            f"Shell n_e={metadata['shell_n_e_cm3']:.1e} cm^-3\n"
            f"Geom/Instr: Core={metadata.get('core_thickness_m', 0.0) * 1e3:.2f} mm, "
            f"Shell={metadata.get('shell_thickness_m', 0.0) * 1e3:.2f} mm, "
            f"FWHM={metadata.get('instrument_fwhm_nm', 0.0):.3f} nm, "
            f"tau_shell_max={metadata['tau_shell_max']:.3f}"
        )

    def show_placeholder(self, message: str = "Click 'Run Simulation' to render the spectrum.") -> None:
        self.plot_container.hide()
        self.message_label.setText(f"CR-LIBS Plot Area\n\n{message}")
        self.message_label.show()

    def plot_spectrum(self, wavelengths: np.ndarray, intensity: np.ndarray, metadata: dict, show_labels: bool) -> None:
        self.data_wl_full = wavelengths
        self.data_inten_full = intensity
        self._last_metadata = metadata
        self._last_show_labels = show_labels
        wavelengths_ds, intensity_ds = self._downsample_for_display(wavelengths, intensity)
        self.message_label.hide()
        self.plot_container.show()

        self.curve_full.setData(wavelengths_ds, intensity_ds)
        self.plot_full.setTitle("Two-Zone CR-LIBS Spectrum", color="#1e293b", size="14pt")
        if intensity_ds.size > 0:
            self.plot_full.setLimits(
                xMin=float(np.min(wavelengths_ds)),
                xMax=float(np.max(wavelengths_ds)),
            )
            current_region = self.region.getRegion()
            wl_min = float(np.min(wavelengths_ds))
            wl_max = float(np.max(wavelengths_ds))
            if current_region[0] < wl_min or current_region[1] > wl_max:
                span = min(100.0, wl_max - wl_min)
                self.region.setRegion((wl_min, wl_min + span))

        self.summary_label.setText(self._format_summary(metadata))
        self._update_roi_plot()

    def _downsample_for_display(self, wavelengths: np.ndarray, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_points = len(wavelengths)
        if n_points <= self.MAX_DISPLAY_POINTS:
            return wavelengths, intensity

        step = int(np.ceil(n_points / self.MAX_DISPLAY_POINTS))
        indices = np.arange(0, n_points, step, dtype=int)
        if indices[-1] != n_points - 1:
            indices = np.append(indices, n_points - 1)
        return wavelengths[indices], intensity[indices]

    def _update_roi_plot(self) -> None:
        if self.data_wl_full is None or self.data_inten_full is None:
            return

        for item in self._roi_markers:
            self.plot_roi.removeItem(item)
        for item in self._roi_labels:
            self.plot_roi.removeItem(item)
        self._roi_markers.clear()
        self._roi_labels.clear()

        r0, r1 = self.region.getRegion()
        mask = (self.data_wl_full >= r0) & (self.data_wl_full <= r1)
        chunk_x = self.data_wl_full[mask]
        chunk_y = self.data_inten_full[mask]

        if chunk_x.size == 0:
            self.curve_roi.setData([], [])
            self.coord_label.setText("ROI: empty")
            return

        self.curve_roi.setData(chunk_x, chunk_y)
        self.plot_roi.setXRange(float(chunk_x[0]), float(chunk_x[-1]), padding=0.02)
        y_min = float(np.min(chunk_y))
        y_max = float(np.max(chunk_y))
        peak_idx = int(np.argmax(chunk_y))
        peak_wl = float(chunk_x[peak_idx])
        peak_int = float(chunk_y[peak_idx])
        if y_max > y_min:
            self.plot_roi.setYRange(y_min, y_max, padding=0.1)
        coord_text = (
            f"ROI: {r0:.2f} - {r1:.2f} nm | "
            f"peak {peak_wl:.2f} nm @ {peak_int:.3e}"
        )

        if not self._last_metadata:
            self.coord_label.setText(coord_text)
            return

        lines_in_roi = [
            line for line in self._last_metadata.get("top_lines", [])
            if r0 <= float(line["wl"]) <= r1
        ]
        if lines_in_roi:
            dominant = max(lines_in_roi, key=lambda line: float(line["int"]))
            coord_text += f" | dominant {dominant['elem']} {dominant['ion']} ({dominant.get('zone', 'Zone')})"
        self.coord_label.setText(coord_text)

        if not self._last_show_labels:
            return

        for line in lines_in_roi[:20]:
            wl = float(line["wl"])
            idx = int(np.argmin(np.abs(chunk_x - wl)))
            y = float(chunk_y[idx])
            marker = pg.ScatterPlotItem([wl], [y], size=7, brush=pg.mkBrush("#f97316"), pen=pg.mkPen(None))
            label = pg.TextItem(
                text=f"{line['elem']} {line['ion']} [{line.get('zone', 'Zone')}]\n{wl:.2f} nm",
                color="#7c2d12",
                anchor=(0.5, 1.0),
            )
            label.setPos(wl, y)
            self.plot_roi.addItem(marker)
            self.plot_roi.addItem(label)
            self._roi_markers.append(marker)
            self._roi_labels.append(label)


class ElementRow(QWidget):
    def __init__(self, on_remove) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.element_combo = QComboBox()
        self.element_combo.addItems(BASE_ELEMENTS)

        self.percentage_spin = QDoubleSpinBox()
        self.percentage_spin.setRange(0.0, 100.0)
        self.percentage_spin.setDecimals(2)
        self.percentage_spin.setSuffix(" %")

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: on_remove(self))

        layout.addWidget(QLabel("Element"))
        layout.addWidget(self.element_combo)
        layout.addWidget(QLabel("Percent"))
        layout.addWidget(self.percentage_spin)
        layout.addWidget(remove_button)

    def get_value(self) -> Tuple[str, float]:
        return self.element_combo.currentText(), float(self.percentage_spin.value())


class LIBSMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CR-LIBS PySide Simulator")
        self.resize(1400, 860)

        self.fetcher = DataFetcher()
        self.canvas = SpectrumCanvas()
        self.element_rows: List[ElementRow] = []

        central = QWidget()
        root = QHBoxLayout(central)
        root.addWidget(self._build_controls())
        root.addWidget(self._build_plot_panel())
        root.setStretch(0, 0)
        root.setStretch(1, 1)
        self.setCentralWidget(central)

        self._add_element_row("Cu", 85.0)
        self._add_element_row("N", 10.0)
        self._add_element_row("O", 4.0)
        self._add_element_row("C", 1.0)

    def _build_plot_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)
        return panel

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        sim_group = QGroupBox("Basic Parameters")
        sim_form = QFormLayout(sim_group)

        self.core_temp_spin = QDoubleSpinBox()
        self.core_temp_spin.setRange(3000.0, 50000.0)
        self.core_temp_spin.setValue(9400.0)
        self.core_temp_spin.setDecimals(1)
        self.core_temp_spin.setSuffix(" K")

        self.core_ne_spin = QDoubleSpinBox()
        self.core_ne_spin.setRange(1e14, 1e19)
        self.core_ne_spin.setValue(1e17)
        self.core_ne_spin.setDecimals(1)
        self.core_ne_spin.setSingleStep(1e16)
        self.core_ne_spin.setSuffix(" cm^-3")
        self.core_ne_spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)

        self.use_rte_checkbox = QCheckBox("Apply Exact RTE")
        self.use_rte_checkbox.setChecked(True)

        self.show_labels_checkbox = QCheckBox("Show Peak Labels")
        self.show_labels_checkbox.setChecked(True)

        sim_form.addRow("Core Temperature", self.core_temp_spin)
        sim_form.addRow("Electron Density", self.core_ne_spin)
        sim_form.addRow(self.use_rte_checkbox)
        sim_form.addRow(self.show_labels_checkbox)

        advanced_group = QGroupBox("Advanced Parameters")
        advanced_form = QFormLayout(advanced_group)

        self.core_ion_offset_spin = QDoubleSpinBox()
        self.core_ion_offset_spin.setRange(-10000.0, 10000.0)
        self.core_ion_offset_spin.setValue(-2000.0)
        self.core_ion_offset_spin.setDecimals(1)
        self.core_ion_offset_spin.setSuffix(" K")

        self.shell_temp_factor_spin = QDoubleSpinBox()
        self.shell_temp_factor_spin.setRange(0.1, 2.0)
        self.shell_temp_factor_spin.setValue(0.5)
        self.shell_temp_factor_spin.setDecimals(3)
        self.shell_temp_factor_spin.setSingleStep(0.05)

        self.shell_ion_offset_spin = QDoubleSpinBox()
        self.shell_ion_offset_spin.setRange(-10000.0, 10000.0)
        self.shell_ion_offset_spin.setValue(-1000.0)
        self.shell_ion_offset_spin.setDecimals(1)
        self.shell_ion_offset_spin.setSuffix(" K")

        self.shell_density_factor_spin = QDoubleSpinBox()
        self.shell_density_factor_spin.setRange(0.001, 2.0)
        self.shell_density_factor_spin.setValue(0.1)
        self.shell_density_factor_spin.setDecimals(4)
        self.shell_density_factor_spin.setSingleStep(0.01)

        self.core_thickness_spin = QDoubleSpinBox()
        self.core_thickness_spin.setRange(0.01, 20.0)
        self.core_thickness_spin.setValue(1.0)
        self.core_thickness_spin.setDecimals(3)
        self.core_thickness_spin.setSuffix(" mm")

        self.shell_thickness_spin = QDoubleSpinBox()
        self.shell_thickness_spin.setRange(0.01, 20.0)
        self.shell_thickness_spin.setValue(2.0)
        self.shell_thickness_spin.setDecimals(3)
        self.shell_thickness_spin.setSuffix(" mm")

        self.instrument_fwhm_spin = QDoubleSpinBox()
        self.instrument_fwhm_spin.setRange(0.001, 2.0)
        self.instrument_fwhm_spin.setValue(0.1)
        self.instrument_fwhm_spin.setDecimals(4)
        self.instrument_fwhm_spin.setSuffix(" nm")

        advanced_form.addRow("Core T_i Offset", self.core_ion_offset_spin)
        advanced_form.addRow("Shell T_e Factor", self.shell_temp_factor_spin)
        advanced_form.addRow("Shell T_i Offset", self.shell_ion_offset_spin)
        advanced_form.addRow("Shell n_e Factor", self.shell_density_factor_spin)
        advanced_form.addRow("Core Thickness", self.core_thickness_spin)
        advanced_form.addRow("Shell Thickness", self.shell_thickness_spin)
        advanced_form.addRow("Instrument FWHM", self.instrument_fwhm_spin)

        comp_group = QGroupBox("Composition")
        comp_layout = QVBoxLayout(comp_group)

        self.total_label = QLabel("Total: 0.00 %")
        self.status_label = QLabel("Status: Ready")
        self.rows_container = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_container)
        self.rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.rows_container)

        add_button = QPushButton("Add Element")
        add_button.clicked.connect(lambda: self._add_element_row())

        comp_layout.addWidget(self.total_label)
        comp_layout.addWidget(self.status_label)
        comp_layout.addWidget(scroll)
        comp_layout.addWidget(add_button)

        action_group = QGroupBox("Actions")
        action_layout = QGridLayout(action_group)
        run_button = QPushButton("Run Simulation")
        run_button.clicked.connect(self._run_simulation)

        preset_button = QPushButton("Load Default")
        preset_button.clicked.connect(self._load_default_composition)

        parse_button = QPushButton("Load CLI Preset")
        parse_button.clicked.connect(self._load_cli_default)

        action_layout.addWidget(run_button, 0, 0)
        action_layout.addWidget(preset_button, 0, 1)
        action_layout.addWidget(parse_button, 1, 0, 1, 2)

        layout.addWidget(sim_group)
        layout.addWidget(advanced_group)
        layout.addWidget(comp_group, 1)
        layout.addWidget(action_group)
        layout.addStretch(1)

        return panel

    def _add_element_row(self, element: str = "Cu", percentage: float = 0.0) -> None:
        row = ElementRow(self._remove_element_row)
        row.element_combo.setCurrentText(element)
        row.percentage_spin.setValue(percentage)
        row.percentage_spin.valueChanged.connect(self._update_total)
        self.element_rows.append(row)
        self.rows_layout.addWidget(row)
        self._update_total()

    def _remove_element_row(self, row: ElementRow) -> None:
        if len(self.element_rows) == 1:
            return
        self.element_rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self._update_total()

    def _update_total(self) -> None:
        total = sum(p for _, p in self._selected_elements())
        self.total_label.setText(f"Total: {total:.2f} %")

    def _selected_elements(self) -> List[Tuple[str, float]]:
        return [row.get_value() for row in self.element_rows if row.get_value()[1] > 0.0]

    def _load_default_composition(self) -> None:
        self._reset_rows([("Cu", 85.0), ("N", 10.0), ("O", 4.0), ("C", 1.0)])

    def _load_cli_default(self) -> None:
        parsed = LIBSSimulator.parse_element_input("Cu 85, N 10, O 4, C 1")
        self._reset_rows(parsed)

    def _reset_rows(self, rows: List[Tuple[str, float]]) -> None:
        for row in self.element_rows[:]:
            self._remove_element_row(row)
        for element, percent in rows:
            self._add_element_row(element, percent)
        self._update_total()

    def _run_simulation(self) -> None:
        selected = self._selected_elements()
        if not selected:
            QMessageBox.warning(self, "Invalid Composition", "Add at least one element with percentage > 0.")
            return

        total_pct = sum(p for _, p in selected)
        if abs(total_pct - 100.0) > 1e-6:
            QMessageBox.warning(self, "Invalid Composition", f"Total percentage must be exactly 100%. Current total: {total_pct:.2f}%")
            return

        try:
            self.status_label.setText("Status: Running simulation...")
            QApplication.processEvents()
            wavelengths, intensity, metadata = self._simulate(selected)
        except Exception as exc:
            self.status_label.setText("Status: Failed")
            QMessageBox.critical(self, "Simulation Failed", str(exc))
            return

        self.canvas.plot_spectrum(
            wavelengths,
            intensity,
            metadata,
            show_labels=self.show_labels_checkbox.isChecked(),
        )
        self.status_label.setText("Status: Render complete")

    def _simulate(self, selected_elements: List[Tuple[str, float]]):
        core_temp = float(self.core_temp_spin.value())
        core_ne = float(self.core_ne_spin.value())
        core_ion_offset = float(self.core_ion_offset_spin.value())
        shell_temp_factor = float(self.shell_temp_factor_spin.value())
        shell_ion_offset = float(self.shell_ion_offset_spin.value())
        shell_density_factor = float(self.shell_density_factor_spin.value())
        core_thickness_m = float(self.core_thickness_spin.value()) * 1e-3
        shell_thickness_m = float(self.shell_thickness_spin.value()) * 1e-3
        instrument_fwhm_nm = float(self.instrument_fwhm_spin.value())

        expanded = []
        for elem, pct in selected_elements:
            base_frac = pct / 100.0
            f_neu, f_ion = PhysicsCalculator.compute_saha_ionization_fractions(
                elem, base_frac, core_temp, core_ne, self.fetcher
            )
            if f_neu > 1e-4:
                expanded.append((elem, 1, f_neu))
            if f_ion > 1e-4:
                expanded.append((elem, 2, f_ion))

        if not expanded:
            raise ValueError("No valid neutral or ionized species produced from the selected composition.")

        total_fraction = sum(frac for _, _, frac in expanded)
        expanded = [(sym, sp, frac / total_fraction) for sym, sp, frac in expanded]

        core = PlasmaZoneParams(
            T_e_K=core_temp,
            T_i_K=max(core_temp + core_ion_offset, 3000.0),
            n_e_cm3=core_ne,
            thickness_m=core_thickness_m,
            label="Core",
        )
        shell_temp = core_temp * shell_temp_factor
        shell = PlasmaZoneParams(
            T_e_K=shell_temp,
            T_i_K=max(shell_temp + shell_ion_offset, 3000.0),
            n_e_cm3=core_ne * shell_density_factor,
            thickness_m=shell_thickness_m,
            label="Shell",
        )

        model = TwoZonePlasma(
            core,
            shell,
            expanded,
            fetcher=self.fetcher,
            use_rte=self.use_rte_checkbox.isChecked(),
            instrument_fwhm_nm=instrument_fwhm_nm,
        )
        return model.run()


class GUIApplication:
    @staticmethod
    def main() -> int:
        app = QApplication(sys.argv)
        window = LIBSMainWindow()
        window.show()
        return app.exec()


if __name__ == "__main__":
    raise SystemExit(GUIApplication.main())
