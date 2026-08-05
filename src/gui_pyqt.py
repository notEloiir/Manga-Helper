"""PySide6 GUI for Manga Helper."""

import sys
from pathlib import Path
from functools import partial
from urllib.request import urlopen
from urllib.parse import urlparse
import shutil
import re

from dataclasses import dataclass
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from PySide6 import QtCore, QtGui, QtWidgets
import os

from src.image_cleaner import clean_image
from src.ocr import (
    detect_text_regions,
    sort_text_regions,
    extract_text,
    extract_text_from_region,
    is_vertical_text,
)
from src.text_processor import (
    convert_to_hiragana,
    convert_to_romaji,
    translate,
)


@dataclass
class RegionData:
    bbox: list
    ocr_text: str = ""
    hiragana: str = ""
    romaji: str = ""
    translation: str = ""
    user_text: str = ""
    orientation: str = "horizontal"
    confidence: float | None = None
    to_update: bool = False
    clean_region: bool = True
    # Per-region text settings
    font_family: str = ""
    font_size: int = 24
    text_angle: float = 0.0
    text_center: bool = True
    text_align_center: bool = True
    text_padding: int = -16
    text_color: tuple = (0, 0, 0)
    text_bg_color: tuple = (255, 255, 255)


class MangaHelperGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga Helper GUI")
        self.resize(1200, 800)

        self.current_image_path = None
        self.original_image = None
        self.region_data = []
        self.selected_index = None
        self.unsaved_changes = False
        self.mode = "select"
        self.drag_state = None
        self.canvas_scale = 1.0
        self.current_draw_rect = None

        # Get available fonts and set defaults
        self.available_fonts = self.find_available_fonts()
        self.font_family = self.get_default_font()
        self.font_size = 24
        self.text_center = True
        self.text_align_center = True
        self.text_padding = -16
        self.text_color = (0, 0, 0)  # Black
        self.text_bg_color = (255, 255, 255)  # White

        # Global persistent defaults (last used settings)
        self.defaults_path = Path("config.json")
        self.global_defaults = {}
        self.window_width = 1200
        self.window_height = 800
        self.fullscreen = False
        self.load_defaults()

        if not self.fullscreen:
            self.resize(self.window_width, self.window_height)
        else:
            self.showFullScreen()

        self._ignore_user_text = False
        self._ignore_clean_checkbox = False
        self._ignore_angle_slider = False
        self._ignore_font_combo = False

        # Create tmp directory for downloads
        self.tmp_dir = Path("./tmp")
        self.tmp_dir.mkdir(exist_ok=True)

        self.setup_ui()

    @staticmethod
    def is_manga_font(font_name):
        """Check if a font is suitable for manga/comic text."""
        manga_font_keywords = [
            "cc wild", "comic", "manga", "anime", "smackattack", "action man",
            "badaboom", "komika jam", "f-rotten", "schlop", "wezer", "ames",
        ]
        return any(keyword in font_name.lower() for keyword in manga_font_keywords)

    @staticmethod
    def find_available_fonts():
        """Find all available fonts from system paths and ./fonts/ directory."""
        fonts = {}
        
        # Search system font directories (Linux/macOS)
        system_font_dirs = [
            "/usr/share/fonts/",
            "/usr/local/share/fonts/",
            os.path.expanduser("~/.local/share/fonts/"),
            os.path.expanduser("~/Library/Fonts/"),
        ]

        pat = re.compile(r"\.([a-z0-9]+)$", re.IGNORECASE)  # captures extension
        font_extentions = { "ttf", "otf", "woff", "woff2", "eot" }
        
        # Search system fonts with manga filter
        for font_dir in system_font_dirs:
            font_path = Path(font_dir)
            if font_path.exists():
                for font_file in font_path.rglob("*"):
                    if font_file.is_file() and pat.search(font_file.name):
                        ext = pat.search(font_file.name).group(1).lower()
                        if ext in font_extentions:
                            font_name = font_file.stem
                            # Only include manga-style fonts from system paths
                            if MangaHelperGUI.is_manga_font(font_name) and font_name not in fonts:
                                fonts[font_name] = str(font_file)
        
        # Also check local ./fonts/ directory (no filtering)
        local_fonts_dir = Path("./fonts/")
        if local_fonts_dir.exists():
            for font_file in local_fonts_dir.glob("*"):
                if font_file.is_file() and pat.search(font_file.name):
                    ext = pat.search(font_file.name).group(1).lower()
                    if ext in font_extentions:
                        font_name = font_file.stem
                        # Add all local fonts without filtering
                        if font_name not in fonts:  # Don't override system fonts
                            fonts[font_name] = str(font_file)

        # Default
        fonts["DejaVuSans"] = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        
        return fonts

    @staticmethod
    def get_default_font():
        """Get the default font, preferring CC Wild Words Roman."""
        fonts = MangaHelperGUI.find_available_fonts()
        if "CC Wild Words Roman" in fonts:
            return "CC Wild Words Roman"
        elif len(fonts):
            # Return the first available font (alphabetically)
            return sorted(fonts.keys())[0]
        else:
            # Fallback to system default
            return "DejaVuSans.ttf"

    def resolve_font_path(self, font_name):
        """Resolve a font name to its full file path."""
        if not font_name:
            return None
        if font_name in self.available_fonts:
            return self.available_fonts[font_name]
        # If a direct file path was given, accept it
        p = Path(font_name)
        if p.exists():
            return str(p)
        return None

    def load_defaults(self):
        defaults = {
            "font_family": self.font_family,
            "font_size": self.font_size,
            "text_padding": self.text_padding,
            "text_color": list(self.text_color),
            "text_bg_color": list(self.text_bg_color),
            "window_width": self.window_width,
            "window_height": self.window_height,
            "fullscreen": self.fullscreen,
        }
        try:
            if self.defaults_path.exists():
                data = json.loads(self.defaults_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    defaults.update(data)
        except Exception:
            pass
        # Ensure font exists
        if defaults.get("font_family") not in self.available_fonts:
            if len(self.available_fonts) > 0:
                defaults["font_family"] = sorted(self.available_fonts.keys())[0]

        try:
            self.window_width = max(200, int(defaults.get("window_width", self.window_width)))
            self.window_height = max(200, int(defaults.get("window_height", self.window_height)))
        except Exception:
            self.window_width = 1200
            self.window_height = 800

        self.fullscreen = bool(defaults.get("fullscreen", self.fullscreen))
        # Apply
        self.global_defaults = defaults
        self.font_family = self.global_defaults.get("font_family", self.font_family)
        self.font_size = int(self.global_defaults.get("font_size", self.font_size))
        self.text_center = True
        self.text_align_center = True
        self.text_padding = int(self.global_defaults.get("text_padding", self.text_padding))
        self.text_color = tuple(self.global_defaults.get("text_color", list(self.text_color)))
        self.text_bg_color = tuple(self.global_defaults.get("text_bg_color", list(self.text_bg_color)))

    def save_defaults(self):
        try:
            data = dict(self.global_defaults)
            data["window_width"] = self.window_width
            data["window_height"] = self.window_height
            data["fullscreen"] = self.fullscreen
            # ensure serializable
            data["text_color"] = list(data.get("text_color", self.text_color))
            data["text_bg_color"] = list(data.get("text_bg_color", self.text_bg_color))
            self.defaults_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def update_window_state(self):
        self.fullscreen = self.isFullScreen()
        if not self.fullscreen:
            self.window_width = self.width()
            self.window_height = self.height()

    def resizeEvent(self, event):
        if not self.isFullScreen():
            self.window_width = self.width()
            self.window_height = self.height()
        return super().resizeEvent(event)

    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.WindowStateChange:
            self.fullscreen = bool(self.windowState() & QtCore.Qt.WindowFullScreen)
            if not self.fullscreen:
                self.window_width = self.width()
                self.window_height = self.height()
        return super().changeEvent(event)

    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        toolbar = QtWidgets.QToolBar()
        self.addToolBar(toolbar)

        def add_separator():
            separator_widget = QtWidgets.QWidget()
            separator_widget.setFixedSize(2, 24)
            separator_widget.setStyleSheet("background-color: rgba(255,255,255,0.3); margin: 0 8px;")
            toolbar.addWidget(separator_widget)

        # File actions
        open_action = QtGui.QAction("Open Image File", self)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        open_action = QtGui.QAction("Open Image URL", self)
        open_action.triggered.connect(self.open_image_from_url)
        toolbar.addAction(open_action)

        save_action = QtGui.QAction("Export Image", self)
        save_action.triggered.connect(self.save_image)
        toolbar.addAction(save_action)

        add_separator()

        save_regions_action = QtGui.QAction("Save Regions", self)
        save_regions_action.triggered.connect(self.save_regions)
        toolbar.addAction(save_regions_action)

        load_regions_action = QtGui.QAction("Load Regions", self)
        load_regions_action.triggered.connect(self.load_regions)
        toolbar.addAction(load_regions_action)

        add_separator()

        # Mode actions
        toolbar.addWidget(QtWidgets.QLabel("Mode:"))
        self.select_action = QtGui.QAction("Selection", self)
        self.select_action.setCheckable(True)
        self.select_action.triggered.connect(self.enter_select_mode)
        toolbar.addAction(self.select_action)

        self.add_region_action = QtGui.QAction("Add Region", self)
        self.add_region_action.setCheckable(True)
        self.add_region_action.toggled.connect(self.toggle_add_region_mode)
        toolbar.addAction(self.add_region_action)

        action_group = QtGui.QActionGroup(self)
        action_group.setExclusive(True)
        action_group.addAction(self.add_region_action)
        action_group.addAction(self.select_action)
        self.select_action.setChecked(True)

        main_layout = QtWidgets.QHBoxLayout(central_widget)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self.preview_widget = PreviewWidget(self)
        left_layout.addWidget(self.preview_widget)

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.set_status("Open an image to begin.")

        # Text editing column
        middle_panel = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(4, 4, 4, 4)

        middle_label = QtWidgets.QLabel("Text Editing")
        middle_label_font = middle_label.font()
        middle_label_font.setBold(True)
        middle_label.setFont(middle_label_font)
        middle_layout.addWidget(middle_label)

        text_edit_group = QtWidgets.QGroupBox("Replace Text")
        text_edit_layout = QtWidgets.QVBoxLayout(text_edit_group)
        self.user_text_edit = QtWidgets.QTextEdit(readOnly=False)
        self.user_text_edit.textChanged.connect(lambda: self.on_user_text_changed(self.user_text_edit.toPlainText()))
        text_edit_layout.addWidget(self.user_text_edit)

        self.clean_checkbox = QtWidgets.QCheckBox("Clean region")
        self.clean_checkbox.stateChanged.connect(self.on_clean_checkbox_changed)
        self.clean_checkbox.setEnabled(False)
        text_edit_layout.addWidget(self.clean_checkbox)

        middle_layout.addWidget(text_edit_group)

        settings_group = QtWidgets.QGroupBox("Text Settings")
        settings_layout = QtWidgets.QFormLayout(settings_group)

        self.font_combo = QtWidgets.QComboBox()
        self.font_combo.addItems(sorted(self.available_fonts.keys()))
        self.font_combo.setCurrentText(self.font_family)
        self.font_combo.currentTextChanged.connect(self.on_font_settings_changed)
        self.font_combo.setMaximumWidth(200)
        settings_layout.addRow("Font:", self.font_combo)

        self.font_size_spin = QtWidgets.QSpinBox()
        self.font_size_spin.setRange(8, 512)
        self.font_size_spin.setValue(self.font_size)
        self.font_size_spin.valueChanged.connect(self.on_font_settings_changed)
        settings_layout.addRow("Size:", self.font_size_spin)

        self.center_check = QtWidgets.QCheckBox("Center text")
        self.center_check.setChecked(self.text_center)
        self.center_check.stateChanged.connect(self.on_font_settings_changed)
        settings_layout.addRow(self.center_check)

        self.align_center_check = QtWidgets.QCheckBox("Align center")
        self.align_center_check.setChecked(self.text_align_center)
        self.align_center_check.stateChanged.connect(self.on_font_settings_changed)
        settings_layout.addRow(self.align_center_check)

        self.padding_spin = QtWidgets.QSpinBox()
        self.padding_spin.setRange(-64, 64)
        self.padding_spin.setValue(self.text_padding)
        self.padding_spin.valueChanged.connect(self.on_font_settings_changed)
        settings_layout.addRow("Padding:", self.padding_spin)

        angle_widget = QtWidgets.QWidget()
        angle_layout = QtWidgets.QHBoxLayout(angle_widget)
        angle_layout.setContentsMargins(0, 0, 0, 0)
        self.angle_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.angle_slider.setRange(-90, 90)
        self.angle_slider.setValue(0)
        self.angle_slider.setSingleStep(1)
        self.angle_slider.valueChanged.connect(self.on_angle_changed)
        angle_layout.addWidget(self.angle_slider)
        self.angle_label = QtWidgets.QLabel("0°")
        self.angle_label.setFixedWidth(40)
        angle_layout.addWidget(self.angle_label)
        settings_layout.addRow("Angle:", angle_widget)

        self.text_color_button = QtWidgets.QPushButton()
        self.text_color_button.setFixedHeight(30)
        self.text_color_button.setToolTip("Click to change text color")
        self.text_color_button.clicked.connect(self.on_text_color_clicked)
        self.update_text_color_button()
        settings_layout.addRow("Text Color:", self.text_color_button)

        self.text_bg_color_button = QtWidgets.QPushButton()
        self.text_bg_color_button.setFixedHeight(30)
        self.text_bg_color_button.setToolTip("Click to change text background color")
        self.text_bg_color_button.clicked.connect(self.on_text_bg_color_clicked)
        self.update_text_bg_color_button()
        settings_layout.addRow("BG Color:", self.text_bg_color_button)

        middle_layout.addWidget(settings_group)
        middle_layout.addStretch(1)

        # Region details and controls column
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        right_label = QtWidgets.QLabel("Region Controls")
        right_label_font = right_label.font()
        right_label_font.setBold(True)
        right_label.setFont(right_label_font)
        right_layout.addWidget(right_label)

        right_layout.addWidget(QtWidgets.QLabel("Detected Regions"))
        self.region_list = QtWidgets.QListWidget()
        self.region_list.currentRowChanged.connect(self.on_region_selected)
        right_layout.addWidget(self.region_list)

        region_buttons = QtWidgets.QWidget()
        region_buttons_layout = QtWidgets.QHBoxLayout(region_buttons)
        region_buttons_layout.setContentsMargins(0, 0, 0, 0)

        self.move_up_button = QtWidgets.QPushButton("↑")
        self.move_up_button.setToolTip("Move selected region up")
        self.move_up_button.clicked.connect(self.move_region_up)
        region_buttons_layout.addWidget(self.move_up_button)

        self.move_down_button = QtWidgets.QPushButton("↓")
        self.move_down_button.setToolTip("Move selected region down")
        self.move_down_button.clicked.connect(self.move_region_down)
        region_buttons_layout.addWidget(self.move_down_button)

        self.delete_region_button = QtWidgets.QPushButton("X")
        self.delete_region_button.setToolTip("Delete selected region")
        self.delete_region_button.clicked.connect(self.delete_selected_region)
        region_buttons_layout.addWidget(self.delete_region_button)

        right_layout.addWidget(region_buttons)

        details_group = QtWidgets.QGroupBox("Region Details")
        details_layout = QtWidgets.QVBoxLayout(details_group)
        self.detail_text = QtWidgets.QTextEdit(readOnly=True)
        details_layout.addWidget(self.detail_text)
        right_layout.addWidget(details_group)
        right_layout.addStretch(1)

        main_layout.addWidget(left_panel, stretch=3)
        main_layout.addWidget(middle_panel, stretch=0.5)
        main_layout.addWidget(right_panel, stretch=1)

    def set_status(self, message):
        self.status_bar.showMessage(message)

    def open_image(self):
        if not self.maybe_prompt_save():
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open manga image",
            filter="Image files (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)",
        )
        if not path:
            return

        self.load_image(Path(path))

    def open_image_from_url(self):
        if not self.maybe_prompt_save():
            return

        dialog = QtWidgets.QInputDialog(self)
        dialog.setWindowTitle("Open Image from URL")
        dialog.setLabelText("Enter the image URL:")
        dialog.setMinimumWidth(600)
        dialog.exec()
        
        if not dialog.result() == QtWidgets.QDialog.Accepted:
            return
        
        url = dialog.textValue().strip()
        if not url:
            return

        self.set_status("Downloading image...")
        QtWidgets.QApplication.processEvents()

        # Extract file extension from URL and download to tmp directory
        parsed_url = urlparse(url)
        url_path = parsed_url.path
        file_ext = Path(url_path).suffix if url_path else ".png"
        if not file_ext:
            file_ext = ".png"
        
        temp_path = self.tmp_dir / f"downloaded_image{file_ext}"
        try:
            with urlopen(url) as response:
                with open(temp_path, 'wb') as out_file:
                    out_file.write(response.read())
            self.load_image(temp_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Download Error", f"Failed to download image:\n{exc}")
            if temp_path.exists():
                temp_path.unlink()

    def load_image(self, path):
        try:
            image = Image.open(path).convert("RGBA")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open Error", f"Unable to open image:\n{exc}")
            return

        self.current_image_path = Path(path)
        self.original_image = image
        self.region_data = []
        self.selected_index = None
        self.unsaved_changes = False
        self.mode = "select"
        self.select_action.setChecked(True)
        self.add_region_action.setChecked(False)

        self.set_status("Detecting text regions...")
        QtWidgets.QApplication.processEvents()

        self.detect_and_initialize_regions()

        self.setWindowTitle(f"Manga Helper GUI — {self.current_image_path.name}")
        self.set_status("Loaded image. Click a region in the list or on the preview.")

    # if text_regions is None use self.region_data instead
    def update_region_data(self, text_regions=None, replace_all=True, sort=True):
        if not text_regions and len(self.region_data) == 0:
            QtWidgets.QMessageBox.information(self, "No text found", "No text regions were detected in this image.")
            self.region_data = []
            self.selected_index = None
            self.refresh_ui()
            return
        
        if not sort:
            sorted_regions = text_regions if text_regions is not None else [ (data.bbox, data.ocr_text, data.confidence) for data in self.region_data ]
        elif text_regions is None:
            sorted_regions = sort_text_regions([ (data.bbox, data.ocr_text, data.confidence) for data in self.region_data ])
        else:
            sorted_regions = sort_text_regions(text_regions)
        
        # Decide whether to replace all (new image) or partial update
        update_index = -1
        if not replace_all:
            for index, data in enumerate(self.region_data):
                if data.to_update:
                    update_index = index

        # Reorder region_data to match sorted_regions indexing
        if not replace_all and sort:
            # If partial update, we can try to match them by bbox
            new_region_data = []
            for index, sorted_region in enumerate(sorted_regions):
                matched = False
                for data in self.region_data:
                    if data.bbox == sorted_region[0]:
                        new_region_data.append(data)
                        matched = True
                        if data.to_update:
                            update_index = index
                        break
                if not matched and update_index != -1:
                    # The only unmatched region should be the one we want to update
                    new_region_data.append(self.region_data[update_index])
                    update_index = index

            self.region_data = new_region_data
        
        # Doesn't need to be extracted each time
        if replace_all:
            extracted_texts = extract_text(self.original_image, sorted_regions)
            translations = translate(extracted_texts)
        else:
            extracted_texts = [
                data.ocr_text if not data.to_update else extract_text_from_region(self.original_image, data.bbox)
                for data in self.region_data
            ]
            translations = [
                data.translation if not data.to_update else translate([extracted_texts[i]])[0]
                for i, data in enumerate(self.region_data)
            ]

        if replace_all:
            self.region_data = []
        for index, (region, text, translation) in enumerate(zip(sorted_regions, extracted_texts, translations)):
            if not replace_all and update_index != index:
                continue

            bbox = self.normalize_bbox(region[0])
            orientation = "vertical" if is_vertical_text(bbox) else "horizontal"
            data = RegionData(
                bbox=bbox,
                ocr_text=text,
                hiragana=convert_to_hiragana(text),
                romaji=convert_to_romaji(text),
                translation=translation,
                user_text="",
                orientation=orientation,
                confidence=region[2] if len(region) > 2 else None,
                to_update=False,
                clean_region=True,
                font_family=self.global_defaults.get("font_family", self.font_family),
                font_size=int(self.global_defaults.get("font_size", self.font_size)),
                text_center=bool(self.global_defaults.get("text_center", self.text_center)),
                text_align_center=bool(self.global_defaults.get("text_align_center", self.text_align_center)),
                text_padding=int(self.global_defaults.get("text_padding", self.text_padding)),
                text_color=tuple(self.global_defaults.get("text_color", list(self.text_color))),
                text_bg_color=tuple(self.global_defaults.get("text_bg_color", list(self.text_bg_color))),
            )
            if not replace_all:
                # Preserve existing per-region text settings when partially updating
                try:
                    old = self.region_data[index]
                    data.font_family = getattr(old, "font_family", data.font_family)
                    data.font_size = getattr(old, "font_size", data.font_size)
                    data.text_center = getattr(old, "text_center", data.text_center)
                    data.text_align_center = getattr(old, "text_align_center", data.text_align_center)
                    data.text_padding = getattr(old, "text_padding", data.text_padding)
                    data.text_color = getattr(old, "text_color", data.text_color)
                    data.text_bg_color = getattr(old, "text_bg_color", data.text_bg_color)
                except Exception:
                    pass
                self.region_data[index] = data
            else:
                self.region_data.append(data)

        if self.region_data and (replace_all or update_index != -1):
            self.selected_index = 0 if replace_all else update_index
        self.refresh_ui()

    def detect_and_initialize_regions(self):
        text_regions = detect_text_regions(self.current_image_path)
        self.update_region_data(text_regions)

    def normalize_bbox(self, bbox):
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    def refresh_ui(self):
        self.refresh_region_list()
        self.update_region_details()
        self.preview_widget.update_preview(self.original_image, self.region_data, self.selected_index, self)

    def refresh_region_list(self):
        selected_index = self.selected_index
        self.region_list.clear()
        for index, region in enumerate(self.region_data, start=1):
            label = f"{index}. [{region.orientation[0]}] {region.ocr_text[:20]}"
            self.region_list.addItem(label)

        self.selected_index = selected_index
        if self.selected_index is not None and 0 <= self.selected_index < len(self.region_data):
            self.region_list.setCurrentRow(self.selected_index)

    def update_region_details(self):
        if self.selected_index is None or self.selected_index >= len(self.region_data):
            self.detail_text.setPlainText("No text region selected.")
            self.user_text_edit.setText("")
            self.user_text_edit.setEnabled(False)
            self._ignore_clean_checkbox = True
            self.clean_checkbox.setChecked(False)
            self._ignore_clean_checkbox = False
            self.clean_checkbox.setEnabled(False)
            self._ignore_angle_slider = True
            self.angle_slider.setValue(0)
            self.angle_label.setText("0°")
            self._ignore_angle_slider = False
            self.angle_slider.setEnabled(False)
            self.update_region_control_buttons()
            return

        region = self.region_data[self.selected_index]
        # If region has no replacement text, ensure it uses the global defaults
        if not region.user_text.strip():
            region.font_family = self.global_defaults.get("font_family", region.font_family)
            region.font_size = int(self.global_defaults.get("font_size", region.font_size))
            region.text_center = bool(self.global_defaults.get("text_center", region.text_center))
            region.text_align_center = bool(self.global_defaults.get("text_align_center", region.text_align_center))
            region.text_padding = int(self.global_defaults.get("text_padding", region.text_padding))
            region.text_color = tuple(self.global_defaults.get("text_color", list(region.text_color)))
            region.text_bg_color = tuple(self.global_defaults.get("text_bg_color", list(region.text_bg_color)))
        lines = [
            "Japanese:",
            region.ocr_text,
            "Hiragana:",
            region.hiragana,
            "Romaji:",
            region.romaji,
            "English:",
            region.translation,
        ]
        self.detail_text.setPlainText("\n".join(lines))
        self.user_text_edit.setEnabled(True)
        self._ignore_user_text = True
        self.user_text_edit.setText(region.user_text)
        self._ignore_user_text = False

        # Populate font and layout controls for this region
        self._ignore_font_combo = True
        try:
            self.font_combo.setCurrentText(region.font_family or self.font_family)
        except Exception:
            pass
        self.font_size_spin.setValue(int(getattr(region, "font_size", self.font_size)))
        self.center_check.setChecked(bool(getattr(region, "text_center", self.text_center)))
        self.align_center_check.setChecked(bool(getattr(region, "text_align_center", self.text_align_center)))
        self.padding_spin.setValue(int(getattr(region, "text_padding", self.text_padding)))
        self._ignore_font_combo = False

        # Update color buttons to reflect region settings
        self.text_color = tuple(getattr(region, "text_color", self.text_color))
        self.update_text_color_button()
        self.text_bg_color = tuple(getattr(region, "text_bg_color", self.text_bg_color))
        self.update_text_bg_color_button()

        self._ignore_clean_checkbox = True
        self.clean_checkbox.setChecked(region.clean_region)
        self._ignore_clean_checkbox = False
        self.clean_checkbox.setEnabled(True)

        self._ignore_angle_slider = True
        self.angle_slider.setValue(int(region.text_angle))
        self.angle_label.setText(f"{int(region.text_angle)}°")
        self._ignore_angle_slider = False
        self.angle_slider.setEnabled(True)

        self.update_region_control_buttons()

    def update_region_control_buttons(self):
        has_selection = self.selected_index is not None and 0 <= self.selected_index < len(self.region_data)
        self.move_up_button.setEnabled(has_selection and self.selected_index > 0)
        self.move_down_button.setEnabled(has_selection and self.selected_index < len(self.region_data) - 1)
        self.delete_region_button.setEnabled(has_selection)

    def move_region_up(self):
        if self.selected_index is None or self.selected_index <= 0:
            return
        self.region_data[self.selected_index - 1], self.region_data[self.selected_index] = (
            self.region_data[self.selected_index],
            self.region_data[self.selected_index - 1],
        )
        self.selected_index -= 1
        self.unsaved_changes = True
        self.update_region_data(replace_all=False, sort=False)
        self.refresh_ui()

    def move_region_down(self):
        if self.selected_index is None or self.selected_index >= len(self.region_data) - 1:
            return
        self.region_data[self.selected_index + 1], self.region_data[self.selected_index] = (
            self.region_data[self.selected_index],
            self.region_data[self.selected_index + 1],
        )
        self.selected_index += 1
        self.unsaved_changes = True
        self.update_region_data(replace_all=False, sort=False)
        self.refresh_ui()

    def delete_selected_region(self):
        if self.selected_index is None or self.selected_index < 0 or self.selected_index >= len(self.region_data):
            return
        del self.region_data[self.selected_index]
        self.unsaved_changes = True
        if self.selected_index >= len(self.region_data):
            self.selected_index = len(self.region_data) - 1 if self.region_data else None
        self.update_region_data(replace_all=False, sort=False)
        self.refresh_ui()

    def on_clean_checkbox_changed(self, state):
        if self._ignore_clean_checkbox or self.selected_index is None:
            return
        self.region_data[self.selected_index].clean_region = bool(state)
        self.unsaved_changes = True
        self.preview_widget.update_preview(self.original_image, self.region_data, self.selected_index, self)

    def enter_add_region_mode(self):
        if self.original_image is None:
            QtWidgets.QMessageBox.information(self, "Open Image", "Please open an image before adding a new region.")
            self.add_region_action.setChecked(False)
            return
        self.mode = "new_region"
        self.add_region_action.setChecked(True)
        self.set_status("Drag on the preview to draw a new text region.")
        self.preview_widget.set_mode(self.mode)

    def enter_select_mode(self):
        self.mode = "select"
        self.select_action.setChecked(True)
        self.set_status("Selection mode enabled.")
        self.preview_widget.set_mode(self.mode)

    def toggle_add_region_mode(self, checked):
        if checked:
            self.mode = "new_region"
            self.set_status("Drag on the preview to draw a new text region.")
        else:
            self.mode = "select"
            self.select_action.setChecked(True)
            self.set_status("Selection mode enabled.")
        self.preview_widget.set_mode(self.mode)

    def on_region_selected(self, row):
        if row < 0:
            self.selected_index = None
        else:
            self.selected_index = row
            if self.mode == "new_region":
                self.enter_select_mode()
        self.update_region_details()
        self.preview_widget.set_selected_index(self.selected_index)
        self.preview_widget.update_preview(self.original_image, self.region_data, self.selected_index, self)

    def on_user_text_changed(self, text):
        if self._ignore_user_text or self.selected_index is None:
            return
        region = self.region_data[self.selected_index]
        # bbox = region.bbox
        # x0, y0 = region.bbox[0]
        # x1, y1 = region.bbox[2]
        # width = abs(x1 - x0)
        # # Textwrap to bbox width
        region.user_text = text
        self.unsaved_changes = True
        self.preview_widget.update_preview(self.original_image, self.region_data, self.selected_index, self)

    def on_font_settings_changed(self):
        if self._ignore_font_combo:
            return
        self.font_family = self.font_combo.currentText()
        self.font_size = self.font_size_spin.value()
        self.text_center = self.center_check.isChecked()
        self.text_align_center = self.align_center_check.isChecked()
        self.text_padding = self.padding_spin.value()

        # Apply to selected region (per-region settings)
        if self.selected_index is not None and 0 <= self.selected_index < len(self.region_data):
            r = self.region_data[self.selected_index]
            r.font_family = self.font_family
            r.font_size = self.font_size
            r.text_center = self.text_center
            r.text_align_center = self.text_align_center
            r.text_padding = self.text_padding
            self.unsaved_changes = True

        # Update global defaults and persist
        self.global_defaults.update({
            "font_family": self.font_family,
            "font_size": self.font_size,
            "text_center": self.text_center,
            "text_align_center": self.text_align_center,
            "text_padding": self.text_padding,
        })
        self.save_defaults()

        self.preview_widget.update_preview(self.original_image, self.region_data, self.selected_index, self)

    def on_angle_changed(self, value):
        self.angle_label.setText(f"{value}°")
        if self._ignore_angle_slider or self.selected_index is None:
            return
        self.region_data[self.selected_index].text_angle = float(value)
        self.unsaved_changes = True
        self.preview_widget.update_preview(self.original_image, self.region_data, self.selected_index, self)

    def on_text_color_clicked(self):
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(*self.text_color),
            self,
            "Select Text Color"
        )
        if color.isValid():
            new_color = (color.red(), color.green(), color.blue())
            self.text_color = new_color
            # update selected region
            if self.selected_index is not None and 0 <= self.selected_index < len(self.region_data):
                self.region_data[self.selected_index].text_color = new_color
                self.unsaved_changes = True
            # persist
            self.global_defaults["text_color"] = list(new_color)
            self.save_defaults()
            self.update_text_color_button()
            self.preview_widget.update_preview(self.original_image, self.region_data, self.selected_index, self)

    def on_text_bg_color_clicked(self):
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(*self.text_bg_color),
            self,
            "Select Text Background Color"
        )
        if color.isValid():
            new_color = (color.red(), color.green(), color.blue())
            self.text_bg_color = new_color
            if self.selected_index is not None and 0 <= self.selected_index < len(self.region_data):
                self.region_data[self.selected_index].text_bg_color = new_color
                self.unsaved_changes = True
            self.global_defaults["text_bg_color"] = list(new_color)
            self.save_defaults()
            self.update_text_bg_color_button()
            self.preview_widget.update_preview(self.original_image, self.region_data, self.selected_index, self)

    def update_text_color_button(self):
        r, g, b = self.text_color
        self.text_color_button.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); color: rgb({255-r}, {255-g}, {255-b});"
        )

    def update_text_bg_color_button(self):
        r, g, b = self.text_bg_color
        self.text_bg_color_button.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); color: rgb({255-r}, {255-g}, {255-b});"
        )

    def create_font(self):
        # Backwards-compatible: create font using current global font settings
        return self.create_font_for(self.font_family, self.font_size)

    def create_font_for(self, font_family=None, font_size=None):
        try:
            family = font_family or self.font_family
            size = font_size or self.font_size
            font_path = self.resolve_font_path(family)
            if font_path:
                return ImageFont.truetype(font_path, max(8, int(size)))
        except Exception:
            pass
        return ImageFont.load_default()

    def save_image(self):
        if self.original_image is None:
            QtWidgets.QMessageBox.information(self, "Export Image", "Open an image first before exporting.")
            return

        dialog = QtWidgets.QFileDialog(self, "Export output image")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setNameFilters(["JPEG image (*.jpg *.jpeg)", "PNG image (*.png)", "All files (*)"])
        default_suffix = self.current_image_path.suffix.lower() if self.current_image_path else "png"
        dialog.setDefaultSuffix(default_suffix)
        if dialog.exec():
            path = dialog.selectedFiles()[0]
        else:
            return

        if not path:
            return

        # Apply the modifications to original image now
        image_to_save = clean_image(
            self.original_image,
            [region.bbox for region in self.region_data if region.clean_region],
            opacity=0.9,
        )
        image_to_save = self.render_user_text_overlay(image_to_save)

        output_path = Path(path)
        if output_path.suffix.lower() in [".jpg", ".jpeg"]:
            background = Image.new("RGB", image_to_save.size, (255, 255, 255))
            if image_to_save.mode == "RGBA":
                background.paste(image_to_save, mask=image_to_save.split()[3])
            else:
                background.paste(image_to_save)
            background.save(output_path, quality=95)
        else:
            image_to_save.save(output_path)

        self.unsaved_changes = False
        self.set_status(f"Saved cleaned image and text export to {output_path.name}")

    def save_regions(self):
        if self.current_image_path is None:
            QtWidgets.QMessageBox.information(self, "Save Regions", "Open an image first before saving regions.")
            return

        dialog = QtWidgets.QFileDialog(self, "Save all regions")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setNameFilters(["JSON file (*.json);;All files (*)"])
        dialog.setDefaultSuffix("json")
        if dialog.exec():
            path = dialog.selectedFiles()[0]
        else:
            return

        if not path:
            return
        export_path = Path(path)

        export_data = {
            "image": self.current_image_path.name,
            "region_count": len(self.region_data),
            "regions": [
                {
                    "bbox": region.bbox,
                    "ocr_text": region.ocr_text,
                    "hiragana": region.hiragana,
                    "romaji": region.romaji,
                    "translation": region.translation,
                    "user_text": region.user_text,
                    "orientation": region.orientation,
                    "confidence": region.confidence,
                    "to_update": region.to_update,
                    "clean_region": region.clean_region,
                    "font_family": region.font_family,
                    "font_size": region.font_size,
                    "text_angle": region.text_angle,
                    "text_center": region.text_center,
                    "text_align_center": region.text_align_center,
                    "text_padding": region.text_padding,
                    "text_color": list(region.text_color) if isinstance(region.text_color, tuple) else region.text_color,
                    "text_bg_color": list(region.text_bg_color) if isinstance(region.text_bg_color, tuple) else region.text_bg_color,
                }
                for region in self.region_data
            ],
        }

        with export_path.open("w", encoding="utf-8") as handle:
            json.dump(export_data, handle, ensure_ascii=False, indent=2)

        self.set_status(f"Regions exported to {export_path.name}")

    def load_regions(self):
        if self.current_image_path is None:
            QtWidgets.QMessageBox.information(self, "Load Regions", "Open an image first before loading regions.")
            return
        if not self.maybe_prompt_save():
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load regions from JSON",
            filter="JSON files (*.json);;All files (*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load regions:\n{exc}")
            return

        # if data.get("image") != self.current_image_path.name:
        #     QtWidgets.QMessageBox.warning(
        #         self,
        #         "Image Mismatch",
        #         f"The regions in {Path(path).name} do not match the currently loaded image.",
        #     )
        #     return

        loaded_regions = []
        for region in data.get("regions", []):
            loaded_regions.append(
                RegionData(
                    bbox=region["bbox"],
                    ocr_text=region["ocr_text"],
                    hiragana=region["hiragana"],
                    romaji=region["romaji"],
                    translation=region["translation"],
                    user_text=region.get("user_text", ""),
                    orientation=region["orientation"],
                    confidence=region.get("confidence"),
                    to_update=region.get("to_update", False),
                    clean_region=region.get("clean_region", True),
                    font_family=region.get("font_family", self.font_family),
                    font_size=int(region.get("font_size", self.font_size)),
                    text_angle=float(region.get("text_angle", 0.0)),
                    text_center=bool(region.get("text_center", self.text_center)),
                    text_align_center=bool(region.get("text_align_center", self.text_align_center)),
                    text_padding=int(region.get("text_padding", self.text_padding)),
                    text_color=tuple(region.get("text_color", list(self.text_color))),
                    text_bg_color=tuple(region.get("text_bg_color", list(self.text_bg_color))),
                )
            )

        self.region_data = loaded_regions
        self.selected_index = 0 if self.region_data else None
        self.unsaved_changes = False
        self.update_region_data(replace_all=False, sort=False)
        self.refresh_ui()

    def maybe_prompt_save(self):
        if not self.unsaved_changes:
            return True
        answer = QtWidgets.QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Save before continuing?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
        )
        if answer == QtWidgets.QMessageBox.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.Yes:
            self.save_image()
        return True

    def closeEvent(self, event):
        if self.maybe_prompt_save():
            self.update_window_state()
            self.save_defaults()
            # Clean up tmp directory
            if self.tmp_dir.exists():
                shutil.rmtree(self.tmp_dir)
            event.accept()
        else:
            event.ignore()

    def render_user_text_overlay(self, image, text_bg_dilation=2):
        draw = ImageDraw.Draw(image)
        for region in self.region_data:
            text = region.user_text.strip()
            if not text:
                continue

            x0, y0 = region.bbox[0]
            x1, y1 = region.bbox[2]
            padding = region.text_padding if hasattr(region, "text_padding") else self.text_padding
            box = (x0 + padding, y0 + padding, x1 - padding, y1 - padding)
            indent = 5
            indent_ratio = 1.1

            # Per-region font
            font = self.create_font_for(getattr(region, "font_family", None), getattr(region, "font_size", None))

            # Check if horizontal center alignment is enabled
            align_center = getattr(region, "text_align_center", self.text_align_center)
            
            # Split text into lines
            lines = text.split('\n')
            
            if align_center and len(lines) > 1:
                # Calculate the width of each line and find max width
                # Also get standard line height from a reference line
                line_bboxes = []
                for line in lines:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_bboxes.append(bbox)
                
                line_widths = [bbox[2] - bbox[0] for bbox in line_bboxes]
                line_heights = [int((bbox[3] - bbox[1]) * indent_ratio + indent) for bbox in line_bboxes]
                
                # Get standard line height from a non-empty line, or use space as reference
                standard_line_height = 0
                for height in line_heights:
                    if height > 0:
                        standard_line_height = height
                        break
                
                # Use standard line height for all lines (including empty ones)
                adjusted_heights = [standard_line_height if h == 0 else h for h in line_heights]
                
                max_width = max(line_widths) if line_widths else 0
                total_height = sum(adjusted_heights)
                
                # Add spacing between lines (1 pixel per line gap)
                if len(lines) > 1:
                    total_height += (len(lines) - 1) * 1
                
                # Create text image with max_width
                text_image = Image.new("RGBA", (max_width + 4, total_height + 4), (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(text_image)
                tc = tuple(getattr(region, "text_color", self.text_color))
                text_color_with_alpha = tc + (255,)
                
                # Draw each line centered within max_width
                y_pos = 2
                for line, adjusted_height in zip(lines, adjusted_heights):
                    line_bbox = draw.textbbox((0, 0), line, font=font)
                    line_width = line_bbox[2] - line_bbox[0]
                    # Center this line within max_width
                    x_pos = 2 + (max_width - line_width) // 2
                    temp_draw.text((x_pos, y_pos), line, fill=text_color_with_alpha, font=font)
                    y_pos += adjusted_height + 1
            else:
                # Single line or align_center disabled: use original behavior
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = int((text_bbox[3] - text_bbox[1]) * indent_ratio + indent)
                text_image = Image.new("RGBA", (text_width + 4, text_height + 4), (0, 0, 0, 0))
                temp_draw = ImageDraw.Draw(text_image)
                tc = tuple(getattr(region, "text_color", self.text_color))
                text_color_with_alpha = tc + (255,)
                temp_draw.text((2, 2), text, fill=text_color_with_alpha, font=font)

            angle = float(region.text_angle)
            if angle != 0.0:
                rotated_text = text_image.rotate(angle, resample=Image.BICUBIC, expand=True)
            else:
                rotated_text = text_image

            rotated_mask = rotated_text.split()[3]
            rotated_width, rotated_height = rotated_text.size

            center_flag = getattr(region, "text_center", self.text_center)
            if center_flag:
                px = box[0] + max(0, (box[2] - box[0] - rotated_width) // 2)
                py = box[1] + max(0, (box[3] - box[1] - rotated_height) // 2)
            else:
                px = box[0]
                py = box[1]

            # Draw background behind rotated text using dilated alpha mask.
            dilation_filter = ImageFilter.MaxFilter(1 + 2 * text_bg_dilation)
            dilated_mask = rotated_mask.filter(dilation_filter)
            bg = tuple(getattr(region, "text_bg_color", self.text_bg_color))
            bg_color_with_alpha = bg + (255,)
            background_overlay = Image.new("RGBA", rotated_text.size, bg_color_with_alpha)
            image.paste(background_overlay, (px, py), dilated_mask)
            image.paste(rotated_text, (px, py), rotated_text)

        return image

    def region_bbox_from_points(self, start, end):
        """Create a normalized bbox from two points."""
        x0, y0 = start
        x1, y1 = end
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    def add_new_region(self, bbox):
        """Add a new text region with the given bbox."""
        defaults = self.global_defaults or {}
        self.region_data.append(RegionData(
            bbox=bbox,
            ocr_text="",
            hiragana="",
            romaji="",
            translation="",
            user_text="",
            orientation="horizontal",
            confidence=None,
            to_update=True,
            clean_region=True,
            font_family=defaults.get("font_family", self.font_family),
            font_size=int(defaults.get("font_size", self.font_size)),
            text_center=True,
            text_align_center=True,
            text_padding=int(defaults.get("text_padding", self.text_padding)),
            text_color=tuple(defaults.get("text_color", list(self.text_color))),
            text_bg_color=tuple(defaults.get("text_bg_color", list(self.text_bg_color))),
            ))
        self.unsaved_changes = True
        self.selected_index = len(self.region_data) - 1
        self.update_region_data(replace_all=False)
        self.refresh_ui()

    def update_region_bbox(self, index, bbox):
        """Update the bbox of a region."""
        if 0 <= index < len(self.region_data):
            self.region_data[index].bbox = bbox
            self.unsaved_changes = True
            self.refresh_ui()


class PreviewWidget(QtWidgets.QLabel):
    EDGE_THRESHOLD = 10  # pixels for edge detection

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #222222;")
        self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.ArrowCursor)
        self.current_image = None
        self.selected_index = None
        self.mode = "select"
        self.start_point = None
        self.preview_rect = None
        self.parent_window = parent
        self.dragging_edge = None
        self.hovered_edge = None
        self.drag_preview_bbox = None

    def set_mode(self, mode):
        self.mode = mode

    def set_selected_index(self, index):
        self.selected_index = index

    def update_preview(self, image, region_data, selected_index, parent_window):
        self.parent_window = parent_window
        self.drag_preview_bbox = None
        if image is None:
            self.clear()
            self.current_image = None
            return

        preview = clean_image(
            image,
            [region.bbox for region in region_data if region.clean_region],
            opacity=0.9,
        )
        preview = parent_window.render_user_text_overlay(preview)

        self.current_image = preview
        self.selected_index = selected_index
        self.setPixmap(self.pillow_to_pixmap(preview))

    def pillow_to_pixmap(self, image):
        data = image.convert("RGBA").tobytes("raw", "RGBA")
        qimage = QtGui.QImage(data, image.width, image.height, QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    def resizeEvent(self, event):
        if self.current_image is not None:
            self.setPixmap(self.pillow_to_pixmap(self.current_image))

    def mousePressEvent(self, event):
        if self.current_image is None:
            return
        
        if self.mode == "new_region":
            self.start_point = event.position()
            self.preview_rect = QtCore.QRect(self.start_point.toPoint(), QtCore.QSize())
        elif self.mode == "select":
            img_coords = self.widget_to_image_coords(event.position())
            resizing = False
            if self.selected_index is not None and img_coords:
                # Check if clicking on an edge of the selected region
                edge = self.detect_edge(img_coords)
                if edge:
                    resizing = True
                    self.dragging_edge = edge
                    self.start_point = event.position()
            
            if not resizing and img_coords:
                # Check if clicking on any region to select it
                for index, region in enumerate(self.parent_window.region_data):
                    bbox = region.bbox
                    x0, y0 = bbox[0]
                    x1, y1 = bbox[2]
                    if x0 <= img_coords[0] <= x1 and y0 <= img_coords[1] <= y1:
                        self.parent_window.selected_index = index
                        self.set_selected_index(index)
                        self.parent_window.refresh_ui()
                        break

    def detect_edge(self, img_coords):
        """Detect which edge or corner of the selected bbox is being hovered over."""
        if self.selected_index is None or self.selected_index >= len(self.parent_window.region_data):
            return None
        
        bbox = self.parent_window.region_data[self.selected_index].bbox
        x, y = img_coords
        x0, y0 = bbox[0]  # top-left
        x1, y1 = bbox[2]  # bottom-right
        
        threshold = self.EDGE_THRESHOLD
        
        # Check for corner hits first (higher priority)
        if abs(x - x0) <= threshold and abs(y - y0) <= threshold:
            return "top-left"
        if abs(x - x1) <= threshold and abs(y - y0) <= threshold:
            return "top-right"
        if abs(x - x1) <= threshold and abs(y - y1) <= threshold:
            return "bottom-right"
        if abs(x - x0) <= threshold and abs(y - y1) <= threshold:
            return "bottom-left"
        
        # Check for edge hits
        if abs(x - x0) <= threshold and y0 <= y <= y1:
            return "left"
        if abs(x - x1) <= threshold and y0 <= y <= y1:
            return "right"
        if abs(y - y0) <= threshold and x0 <= x <= x1:
            return "top"
        if abs(y - y1) <= threshold and x0 <= x <= x1:
            return "bottom"
        
        return None

    def get_cursor_for_edge(self, edge):
        """Get the appropriate cursor for a given edge."""
        cursor_map = {
            "left": QtCore.Qt.SizeHorCursor,
            "right": QtCore.Qt.SizeHorCursor,
            "top": QtCore.Qt.SizeVerCursor,
            "bottom": QtCore.Qt.SizeVerCursor,
            "top-left": QtCore.Qt.SizeFDiagCursor,
            "top-right": QtCore.Qt.SizeBDiagCursor,
            "bottom-left": QtCore.Qt.SizeBDiagCursor,
            "bottom-right": QtCore.Qt.SizeFDiagCursor,
        }
        return cursor_map.get(edge, QtCore.Qt.ArrowCursor)

    def update_cursor_for_current_position(self, event):
        if self.current_image is None or self.mode != "select":
            return
        img_coords = self.widget_to_image_coords(event.position())
        if img_coords:
            edge = self.detect_edge(img_coords)
            self.hovered_edge = edge
            if edge:
                self.setCursor(self.get_cursor_for_edge(edge))
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)
        else:
            self.hovered_edge = None
            self.setCursor(QtCore.Qt.ArrowCursor)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if self.mode == "select" and not self.dragging_edge:
            self.hovered_edge = None
            self.setCursor(QtCore.Qt.ArrowCursor)

    def mouseMoveEvent(self, event):
        if self.current_image is None:
            return
        
        if self.mode == "new_region" and self.start_point is not None:
            current_point = event.position()
            self.preview_rect = QtCore.QRect(self.start_point.toPoint(), current_point.toPoint()).normalized()
            self.update()
        elif self.mode == "select":
            if self.dragging_edge:
                # Handle dragging to resize
                img_coords = self.widget_to_image_coords(event.position())
                if img_coords:
                    self.resize_bbox_by_edge(img_coords)
                    self.setCursor(self.get_cursor_for_edge(self.dragging_edge))
            else:
                self.update_cursor_for_current_position(event)

    def resize_bbox_by_edge(self, new_coords):
        """Resize the bbox by dragging an edge."""
        if self.selected_index is None or not self.parent_window:
            return None

        # snapshot index
        index = self.selected_index
        region = self.parent_window.region_data[index]
        bbox = region.bbox
        x0, y0 = bbox[0]
        x1, y1 = bbox[2]
        new_x, new_y = new_coords

        edge = self.dragging_edge

        if edge == "left":
            x0 = new_x
        elif edge == "right":
            x1 = new_x
        elif edge == "top":
            y0 = new_y
        elif edge == "bottom":
            y1 = new_y
        elif edge == "top-left":
            x0, y0 = new_x, new_y
        elif edge == "top-right":
            x1, y1 = new_x, new_y
        elif edge == "bottom-left":
            x0, y1 = new_x, new_y
        elif edge == "bottom-right":
            x1, y1 = new_x, new_y

        min_size = 20
        if abs(x1 - x0) >= min_size and abs(y1 - y0) >= min_size:
            new_bbox = [
                (min(x0, x1), min(y0, y1)),
                (max(x0, x1), min(y0, y1)),
                (max(x0, x1), max(y0, y1)),
                (min(x0, x1), max(y0, y1)),
            ]
            self.drag_preview_bbox = new_bbox
            self.update()
            self.selected_index = index
            return new_bbox

        self.drag_preview_bbox = None
        self.update()
        return None

    def mouseReleaseEvent(self, event):
        if self.current_image is None:
            return
        
        if self.mode == "new_region" and self.start_point is not None:
            end_point = event.position()
            if self.parent_window is None:
                return
            img_start = self.widget_to_image_coords(self.start_point)
            img_end = self.widget_to_image_coords(end_point)
            if img_start is None or img_end is None:
                self.start_point = None
                self.preview_rect = None
                self.update()
                return
            bbox = self.parent_window.region_bbox_from_points(img_start, img_end)
            for existing_region in self.parent_window.region_data:
                existing_bbox = existing_region.bbox
                if bbox == existing_bbox:
                    QtWidgets.QMessageBox.information(self, "Duplicate Region", "This region already exists.")
                    self.start_point = None
                    self.preview_rect = None
                    self.update()
                    return
            self.parent_window.add_new_region(bbox)
            self.parent_window.enter_select_mode()
            self.start_point = None
            self.preview_rect = None
            self.update()
        elif self.mode == "select":

            if self.dragging_edge:
                img_coords = self.widget_to_image_coords(event.position())
                if not img_coords:
                    self.dragging_edge = None
                    self.drag_preview_bbox = None
                    self.update()
                    return
                new_bbox = self.resize_bbox_by_edge(img_coords)
                if new_bbox is not None:
                    self.parent_window.update_region_bbox(self.selected_index, new_bbox)
                    curr_region_data = self.parent_window.region_data[self.selected_index]
                    curr_region_data.to_update = True
                    self.parent_window.update_region_data(replace_all=False)

            self.dragging_edge = None
            self.drag_preview_bbox = None
            self.update_cursor_for_current_position(event)
            self.start_point = None
            self.update()

    def widget_to_image_coords(self, point):
        if self.current_image is None:
            return None
        label_size = self.size()
        image_width, image_height = self.current_image.size
        if image_width == 0 or image_height == 0:
            return None

        scale = min(label_size.width() / image_width, label_size.height() / image_height)
        px = py = 0
        x = (point.x() - px) / scale
        y = (point.y() - py) / scale
        if x < 0 or y < 0 or x > image_width or y > image_height:
            return None
        return (int(x), int(y))

    def image_to_widget_coords(self, point):
        if self.current_image is None:
            return None
        label_size = self.size()
        image_width, image_height = self.current_image.size
        if image_width == 0 or image_height == 0:
            return None
        scale = min(label_size.width() / image_width, label_size.height() / image_height)
        px = py = 0
        x = point[0] * scale + px
        y = point[1] * scale + py
        return (x, y)

    def draw_bbox(self, painter, bbox, color=(0, 255, 0), width=3, dash=False):
        if self.current_image is None:
            return
        x0, y0 = bbox[0]
        x1, y1 = bbox[2]
        top_left = self.image_to_widget_coords((x0, y0))
        bottom_right = self.image_to_widget_coords((x1, y1))
        if top_left is None or bottom_right is None:
            return

        rect = QtCore.QRectF(top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
        pen = QtGui.QPen(QtGui.QColor(*color), width)
        if dash:
            pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(rect)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        if self.preview_rect is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor("cyan"), 2, QtCore.Qt.DashLine))
            painter.drawRect(self.preview_rect)

        if self.mode == "select" and self.current_image is not None:
            if self.drag_preview_bbox is not None:
                self.draw_bbox(painter, self.drag_preview_bbox, color=(255, 255, 0), width=3)
            elif self.selected_index is not None and self.parent_window is not None:
                if 0 <= self.selected_index < len(self.parent_window.region_data):
                    selected_bbox = self.parent_window.region_data[self.selected_index].bbox
                    self.draw_bbox(painter, selected_bbox, color=(0, 255, 0), width=3)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MangaHelperGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
