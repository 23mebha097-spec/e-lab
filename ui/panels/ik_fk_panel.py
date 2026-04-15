from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np


class IKFKPanel(QtWidgets.QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.sliders = {}
        self._joint_order = []
        self.init_ui()

    def init_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        content_scroll = QtWidgets.QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        root.addWidget(content_scroll, 3)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setAlignment(QtCore.Qt.AlignTop)
        content_layout.setSpacing(12)
        content_scroll.setWidget(content)

        title = QtWidgets.QLabel("IK and FK")
        title.setStyleSheet("color: #1565c0; font-size: 24px; font-weight: 700; padding: 4px 6px;")
        content_layout.addWidget(title)

        subtitle = QtWidgets.QLabel("Inverse Kinematics input first, then Forward Kinematics using DH parameters")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #546e7a; font-size: 14px; padding: 0 6px 8px 6px;")
        content_layout.addWidget(subtitle)

        self.ik_group = QtWidgets.QFrame()
        self.ik_group.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #dbe6ee; border-radius: 10px; }"
        )
        ik_layout = QtWidgets.QVBoxLayout(self.ik_group)
        ik_layout.setContentsMargins(12, 12, 12, 12)
        ik_layout.setSpacing(10)

        ik_header = QtWidgets.QLabel("Inverse Kinematics")
        ik_header.setStyleSheet("color: #1976d2; font-size: 19px; font-weight: bold;")
        ik_layout.addWidget(ik_header)

        target_row = QtWidgets.QHBoxLayout()
        self.ik_x = self._make_num_input(-99999, 99999, 0.0, 3)
        self.ik_y = self._make_num_input(-99999, 99999, 0.0, 3)
        self.ik_z = self._make_num_input(-99999, 99999, 0.0, 3)
        target_row.addWidget(self._labeled_widget("Target X", self.ik_x))
        target_row.addWidget(self._labeled_widget("Target Y", self.ik_y))
        target_row.addWidget(self._labeled_widget("Target Z", self.ik_z))
        ik_layout.addLayout(target_row)

        solver_row = QtWidgets.QHBoxLayout()
        self.ik_tolerance = self._make_num_input(1e-6, 1000.0, 0.1, 4)
        self.ik_iters = QtWidgets.QSpinBox()
        self.ik_iters.setRange(10, 5000)
        self.ik_iters.setValue(350)
        self.ik_iters.setStyleSheet(self._spin_style())
        solver_row.addWidget(self._labeled_widget("Tolerance", self.ik_tolerance))
        solver_row.addWidget(self._labeled_widget("Max Iterations", self.ik_iters))

        self.solve_ik_btn = QtWidgets.QPushButton("Solve IK from DH")
        self.solve_ik_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.solve_ik_btn.setStyleSheet(self._primary_btn_style())
        self.solve_ik_btn.clicked.connect(self.solve_ik)
        solver_row.addWidget(self.solve_ik_btn)
        ik_layout.addLayout(solver_row)

        content_layout.addWidget(self.ik_group)

        self.fk_group = QtWidgets.QFrame()
        self.fk_group.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #dbe6ee; border-radius: 10px; }"
        )
        fk_layout = QtWidgets.QVBoxLayout(self.fk_group)
        fk_layout.setContentsMargins(12, 12, 12, 12)
        fk_layout.setSpacing(10)

        fk_header = QtWidgets.QLabel("Forward Kinematics (Standard DH)")
        fk_header.setStyleSheet("color: #1976d2; font-size: 19px; font-weight: bold;")
        fk_layout.addWidget(fk_header)

        tool_row = QtWidgets.QHBoxLayout()
        self.refresh_dh_btn = QtWidgets.QPushButton("Match Rows with Active Joints")
        self.refresh_dh_btn.setStyleSheet(self._ghost_btn_style())
        self.refresh_dh_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.refresh_dh_btn.clicked.connect(self.rebuild_dh_table)
        tool_row.addWidget(self.refresh_dh_btn)

        self.load_theta_btn = QtWidgets.QPushButton("Load Current Joint Angles")
        self.load_theta_btn.setStyleSheet(self._ghost_btn_style())
        self.load_theta_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.load_theta_btn.clicked.connect(self.load_joint_angles_into_dh)
        tool_row.addWidget(self.load_theta_btn)

        self.run_fk_btn = QtWidgets.QPushButton("Compute FK")
        self.run_fk_btn.setStyleSheet(self._primary_btn_style())
        self.run_fk_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.run_fk_btn.clicked.connect(self.update_display)
        tool_row.addWidget(self.run_fk_btn)
        tool_row.addStretch()
        fk_layout.addLayout(tool_row)

        self.dh_table = QtWidgets.QTableWidget(0, 5)
        self.dh_table.setHorizontalHeaderLabels(["Joint", "theta (deg)", "d", "a", "alpha (deg)"])
        self.dh_table.verticalHeader().setVisible(False)
        self.dh_table.setAlternatingRowColors(True)
        self.dh_table.setStyleSheet(
            """
            QTableWidget {
                background: #ffffff;
                color: #102a43;
                border: 1px solid #dbe6ee;
                border-radius: 8px;
                font-size: 14px;
                gridline-color: #e3edf5;
            }
            QHeaderView::section {
                background: #f0f6fb;
                color: #1e3a5f;
                font-size: 14px;
                font-weight: 700;
                border: none;
                border-bottom: 1px solid #dbe6ee;
                padding: 8px;
            }
            """
        )
        self.dh_table.horizontalHeader().setStretchLastSection(True)
        self.dh_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        fk_layout.addWidget(self.dh_table)

        self.result_view = QtWidgets.QTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setMinimumHeight(220)
        self.result_view.setStyleSheet(
            """
            QTextEdit {
                background: #ffffff;
                color: #0f172a;
                border: 1px solid #dbe6ee;
                border-radius: 8px;
                font-size: 14px;
                padding: 10px;
            }
            """
        )
        fk_layout.addWidget(self.result_view)

        content_layout.addWidget(self.fk_group)

        slider_header = QtWidgets.QLabel("Joint Rotation Controls")
        slider_header.setStyleSheet("color: #1976d2; font-size: 20px; font-weight: 700; padding: 4px;")
        root.addWidget(slider_header)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #ffffff; border: 1px solid #dbe6ee; border-radius: 10px;")

        self.slider_container = QtWidgets.QWidget()
        self.slider_layout = QtWidgets.QVBoxLayout(self.slider_container)
        self.slider_layout.setAlignment(QtCore.Qt.AlignTop)
        self.scroll_area.setWidget(self.slider_container)

        root.addWidget(self.scroll_area, 2)

        self.rebuild_dh_table()
        self.refresh_sliders()
        self.update_display()

    def _labeled_widget(self, text, widget):
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("QFrame { background: transparent; border: none; }")
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color: #455a64; font-size: 14px; font-weight: 600;")
        lay.addWidget(lbl)
        lay.addWidget(widget)
        return frame

    def _make_num_input(self, lo, hi, val, decimals):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setValue(val)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        spin.setStyleSheet(self._spin_style())
        return spin

    def _spin_style(self):
        return (
            "QDoubleSpinBox, QSpinBox {"
            " background: #ffffff;"
            " color: #1565c0;"
            " border: 1px solid #90caf9;"
            " border-radius: 6px;"
            " padding: 6px 8px;"
            " font-size: 15px;"
            " font-weight: 700;"
            "}"
        )

    def _primary_btn_style(self):
        return (
            "QPushButton {"
            " background-color: #1976d2;"
            " color: white;"
            " border: none;"
            " border-radius: 8px;"
            " padding: 8px 14px;"
            " font-size: 14px;"
            " font-weight: 700;"
            "}"
            "QPushButton:hover { background-color: #1565c0; }"
        )

    def _ghost_btn_style(self):
        return (
            "QPushButton {"
            " background-color: #ffffff;"
            " color: #1976d2;"
            " border: 1px solid #90caf9;"
            " border-radius: 8px;"
            " padding: 8px 14px;"
            " font-size: 14px;"
            " font-weight: 700;"
            "}"
            "QPushButton:hover { background-color: #e3f2fd; }"
        )

    def _active_joint_child_names(self):
        joint_tab = getattr(self.mw, "joint_tab", None)
        if joint_tab is None:
            return []

        joint_data = getattr(joint_tab, "joints", {})
        result = []
        for child_name, data in joint_data.items():
            joint_id = data.get("joint_id", child_name)
            is_slave = False
            for _, slaves in self.mw.robot.joint_relations.items():
                if any(s_id == joint_id for s_id, _ in slaves):
                    is_slave = True
                    break
            if not is_slave:
                result.append(child_name)
        return result

    def rebuild_dh_table(self):
        joint_tab = getattr(self.mw, "joint_tab", None)
        joint_data = getattr(joint_tab, "joints", {}) if joint_tab is not None else {}

        prev = {}
        for r in range(self.dh_table.rowCount()):
            j_name = self.dh_table.item(r, 0).text() if self.dh_table.item(r, 0) else f"J{r+1}"
            prev[j_name] = [
                self._table_num(r, 1, 0.0),
                self._table_num(r, 2, 0.0),
                self._table_num(r, 3, 0.0),
                self._table_num(r, 4, 0.0),
            ]

        self._joint_order = self._active_joint_child_names()
        if not self._joint_order:
            self._joint_order = [f"J{i+1}" for i in range(max(1, self.dh_table.rowCount()))]

        self.dh_table.setRowCount(len(self._joint_order))
        for idx, child_name in enumerate(self._joint_order):
            display = child_name
            if child_name in joint_data:
                display = joint_data[child_name].get("custom_name", child_name)

            name_item = QtWidgets.QTableWidgetItem(display)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.dh_table.setItem(idx, 0, name_item)

            values = prev.get(display, [0.0, 0.0, 0.0, 0.0])
            for col, val in enumerate(values, start=1):
                self.dh_table.setItem(idx, col, QtWidgets.QTableWidgetItem(f"{val:.4f}"))

        self.load_joint_angles_into_dh()

    def _table_num(self, row, col, default):
        item = self.dh_table.item(row, col)
        if not item:
            return default
        try:
            return float(item.text())
        except Exception:
            return default

    def load_joint_angles_into_dh(self):
        joint_tab = getattr(self.mw, "joint_tab", None)
        if joint_tab is None:
            return

        if not self._joint_order:
            return

        for idx, child_name in enumerate(self._joint_order):
            if child_name not in joint_tab.joints:
                continue
            theta = joint_tab.joints[child_name].get("current_angle", 0.0)
            self.dh_table.setItem(idx, 1, QtWidgets.QTableWidgetItem(f"{theta:.4f}"))

    def _dh_rows(self):
        rows = []
        for r in range(self.dh_table.rowCount()):
            theta_deg = self._table_num(r, 1, 0.0)
            d = self._table_num(r, 2, 0.0)
            a = self._table_num(r, 3, 0.0)
            alpha_deg = self._table_num(r, 4, 0.0)
            rows.append((theta_deg, d, a, alpha_deg))
        return rows

    def _a_matrix(self, theta_rad, d, a, alpha_rad):
        ct, st = np.cos(theta_rad), np.sin(theta_rad)
        ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _fk_with_thetas_rad(self, thetas_rad):
        rows = self._dh_rows()
        T = np.eye(4)
        chain = []
        for i, (_, d, a, alpha_deg) in enumerate(rows):
            alpha_rad = np.radians(alpha_deg)
            A = self._a_matrix(thetas_rad[i], d, a, alpha_rad)
            T = T @ A
            chain.append(T.copy())
        return T, chain

    def update_display(self):
        rows = self._dh_rows()
        if not rows:
            self.result_view.setHtml("<p style='color:#78909c;'>No DH rows configured.</p>")
            return

        theta_rad = np.array([np.radians(r[0]) for r in rows], dtype=float)
        T, chain = self._fk_with_thetas_rad(theta_rad)

        html = """
        <style>
            .box { border: 1px solid #dbe6ee; border-radius: 10px; margin: 10px 0; overflow: hidden; }
            .head { background: #1976d2; color: #fff; font-weight: 700; font-size: 16px; padding: 10px 12px; }
            .matrix { width: 100%; border-collapse: collapse; }
            .matrix td, .matrix th { text-align: center; padding: 8px; font-family: Consolas, monospace; font-size: 14px; }
            .matrix th { background: #f2f7fb; color: #334e68; }
            .matrix tr:nth-child(even) td { background: #fafcfe; }
        </style>
        """

        for i, Ti in enumerate(chain):
            html += f"<div class='box'><div class='head'>T0{i+1}</div>{self._matrix_html(Ti)}</div>"

        html += f"<div class='box'><div class='head'>End-Effector Pose</div>{self._matrix_html(T)}</div>"
        p = T[:3, 3]
        html += (
            f"<p style='font-size:15px;color:#0f172a;'><b>Position:</b> "
            f"X={p[0]:.4f}, Y={p[1]:.4f}, Z={p[2]:.4f}</p>"
        )

        self.result_view.setHtml(html)

    def _matrix_html(self, mat):
        labels = ["X", "Y", "Z", "T"]
        s = "<table class='matrix'><tr>"
        for lbl in labels:
            s += f"<th>{lbl}</th>"
        s += "</tr>"
        for row in mat:
            s += "<tr>"
            for val in row:
                s += f"<td>{val: .5f}</td>"
            s += "</tr>"
        s += "</table>"
        return s

    def solve_ik(self):
        rows = self._dh_rows()
        n = len(rows)
        if n == 0:
            QtWidgets.QMessageBox.warning(self, "IK", "Please configure DH rows first.")
            return

        target = np.array([self.ik_x.value(), self.ik_y.value(), self.ik_z.value()], dtype=float)
        tol = max(1e-8, self.ik_tolerance.value())
        max_iters = self.ik_iters.value()

        theta = np.array([np.radians(r[0]) for r in rows], dtype=float)
        lam = 1e-2
        eps = 1e-5

        def pos_of(th):
            T, _ = self._fk_with_thetas_rad(th)
            return T[:3, 3]

        converged = False
        for _ in range(max_iters):
            p = pos_of(theta)
            err = target - p
            if np.linalg.norm(err) <= tol:
                converged = True
                break

            J = np.zeros((3, n), dtype=float)
            for i in range(n):
                tp = theta.copy()
                tm = theta.copy()
                tp[i] += eps
                tm[i] -= eps
                J[:, i] = (pos_of(tp) - pos_of(tm)) / (2.0 * eps)

            jj_t = J @ J.T
            step = J.T @ np.linalg.solve(jj_t + (lam * lam) * np.eye(3), err)
            step = np.clip(step, -0.35, 0.35)
            theta += step

        theta_deg = np.degrees(theta)
        for i in range(n):
            self.dh_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{theta_deg[i]:.4f}"))

        self.update_display()

        solved_text = "converged" if converged else "best effort"
        self.mw.log(f"DH IK solve complete ({solved_text}).")

    def refresh_sliders(self):
        while self.slider_layout.count():
            item = self.slider_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.sliders = {}

        joint_tab = getattr(self.mw, "joint_tab", None)
        joint_data = getattr(joint_tab, "joints", {}) if joint_tab is not None else {}
        if not joint_data:
            empty_msg = QtWidgets.QLabel("No joints created yet.")
            empty_msg.setStyleSheet("color: #90a4ae; font-size: 14px; font-style: italic; padding: 10px;")
            self.slider_layout.addWidget(empty_msg)
            return

        for child_name, data in joint_data.items():
            joint_id = data.get("joint_id", child_name)
            is_slave = False
            for _, slaves in self.mw.robot.joint_relations.items():
                if any(s_id == joint_id for s_id, _ in slaves):
                    is_slave = True
                    break
            if is_slave:
                continue

            group = QtWidgets.QFrame()
            group.setStyleSheet("background: #ffffff; border: 1px solid #dbe6ee; border-radius: 8px; margin-bottom: 6px;")
            glay = QtWidgets.QVBoxLayout(group)
            glay.setContentsMargins(10, 8, 10, 8)

            custom_name = data.get("custom_name", f"{data['parent']} -> {child_name}")
            axis_name = ["X", "Y", "Z"][data["axis"]]
            lbl = QtWidgets.QLabel(f"{custom_name} ({axis_name})")
            lbl.setStyleSheet("color: #1565c0; font-weight: 700; font-size: 16px;")
            glay.addWidget(lbl)

            row = QtWidgets.QHBoxLayout()
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(int(data["min"] * 10), int(data["max"] * 10))
            slider.setValue(int(data.get("current_angle", 0.0) * 10))
            slider.setCursor(QtCore.Qt.PointingHandCursor)
            slider.setStyleSheet(
                """
                QSlider::groove:horizontal { height: 8px; background: #f0f4f8; border-radius: 4px; border: 1px solid #d3deea; }
                QSlider::sub-page:horizontal { background: #90caf9; border-radius: 4px; }
                QSlider::handle:horizontal { background: white; border: 2px solid #1976d2; width: 16px; height: 16px; margin-top: -5px; margin-bottom: -5px; border-radius: 8px; }
                """
            )

            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(data["min"], data["max"])
            spin.setValue(data.get("current_angle", 0.0))
            spin.setFixedWidth(90)
            spin.setDecimals(2)
            spin.setStyleSheet(self._spin_style())

            slider.valueChanged.connect(lambda v, c=child_name, s=spin: self.on_slider_move(c, v / 10.0, s))
            spin.valueChanged.connect(lambda v, c=child_name, sl=slider: self.on_spin_move(c, v, sl))

            row.addWidget(slider)
            row.addWidget(spin)
            glay.addLayout(row)
            self.slider_layout.addWidget(group)
            self.sliders[child_name] = {"slider": slider, "spin": spin}

    def on_slider_move(self, child_name, value, spinbox):
        spinbox.blockSignals(True)
        spinbox.setValue(value)
        spinbox.blockSignals(False)
        self.apply_rotation(child_name, value)

    def on_spin_move(self, child_name, value, slider):
        slider.blockSignals(True)
        slider.setValue(int(value * 10))
        slider.blockSignals(False)
        self.apply_rotation(child_name, value)

    def apply_rotation(self, child_name, angle):
        joint_tab = getattr(self.mw, "joint_tab", None)
        if joint_tab is None or child_name not in joint_tab.joints:
            return
        joint_tab.apply_joint_rotation(child_name, angle)
        if child_name in self._joint_order:
            row = self._joint_order.index(child_name)
            self.dh_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{angle:.4f}"))
        self.update_display()

    def sync_slider(self, child_name, value):
        if child_name in self.sliders:
            data = self.sliders[child_name]
            data["slider"].blockSignals(True)
            data["slider"].setValue(int(value * 10))
            data["slider"].blockSignals(False)
            data["spin"].blockSignals(True)
            data["spin"].setValue(value)
            data["spin"].blockSignals(False)

        if child_name in self._joint_order:
            row = self._joint_order.index(child_name)
            self.dh_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{value:.4f}"))
        self.update_display()
