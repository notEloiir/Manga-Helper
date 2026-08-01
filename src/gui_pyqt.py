"""PySide6 GUI for Manga Helper."""

import sys
from pathlib import Path
from functools import partial

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from PySide6 import QtCore, QtGui, QtWidgets

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
    translate_with_context,
)


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

        self.font_family = "DejaVuSans.ttf"
        self.font_size = 24
        self.text_center = True
        self.text_padding = -16

        self._ignore_user_text = False

        self.setup_ui()

    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        toolbar = QtWidgets.QToolBar()
        self.addToolBar(toolbar)

        open_action = QtGui.QAction("Open Image", self)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)

        self.add_region_action = QtGui.QAction("Add Region", self)
        self.add_region_action.setCheckable(True)
        self.add_region_action.toggled.connect(self.toggle_add_region_mode)
        toolbar.addAction(self.add_region_action)

        select_action = QtGui.QAction("Selection Mode", self)
        select_action.triggered.connect(self.enter_select_mode)
        toolbar.addAction(select_action)

        save_action = QtGui.QAction("Save Image", self)
        save_action.triggered.connect(self.save_image)
        toolbar.addAction(save_action)

        export_action = QtGui.QAction("Export Text", self)
        export_action.triggered.connect(self.export_text)
        toolbar.addAction(export_action)

        self.mode_label = QtWidgets.QLabel("Mode: Select")
        toolbar.addWidget(self.mode_label)

        main_layout = QtWidgets.QHBoxLayout(central_widget)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self.preview_widget = PreviewWidget(self)
        left_layout.addWidget(self.preview_widget)

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.set_status("Open an image to begin.")

        middle_panel = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(4, 4, 4, 4)

        text_edit_group = QtWidgets.QGroupBox("Replace Text")
        text_edit_layout = QtWidgets.QVBoxLayout(text_edit_group)
        self.user_text_edit = QtWidgets.QTextEdit(readOnly=False)
        self.user_text_edit.textChanged.connect(lambda: self.on_user_text_changed(self.user_text_edit.toPlainText()))
        text_edit_layout.addWidget(self.user_text_edit)
        middle_layout.addWidget(text_edit_group)

        settings_group = QtWidgets.QGroupBox("Text Settings")
        settings_layout = QtWidgets.QFormLayout(settings_group)

        self.font_entry = QtWidgets.QLineEdit(self.font_family)
        settings_layout.addRow("Font name:", self.font_entry)

        self.font_size_spin = QtWidgets.QSpinBox()
        self.font_size_spin.setRange(8, 128)
        self.font_size_spin.setValue(self.font_size)
        self.font_size_spin.valueChanged.connect(self.on_font_settings_changed)
        settings_layout.addRow("Size:", self.font_size_spin)

        self.center_check = QtWidgets.QCheckBox("Center text")
        self.center_check.setChecked(self.text_center)
        self.center_check.stateChanged.connect(self.on_font_settings_changed)
        settings_layout.addRow(self.center_check)

        self.padding_spin = QtWidgets.QSpinBox()
        self.padding_spin.setRange(-64, 64)
        self.padding_spin.setValue(self.text_padding)
        self.padding_spin.valueChanged.connect(self.on_font_settings_changed)
        settings_layout.addRow("Padding:", self.padding_spin)

        middle_layout.addWidget(settings_group)
        middle_layout.addStretch(1)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        right_layout.addWidget(QtWidgets.QLabel("Detected Text Areas"))
        self.region_list = QtWidgets.QListWidget()
        self.region_list.currentRowChanged.connect(self.on_region_selected)
        right_layout.addWidget(self.region_list)

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
        self.update_mode_label()

        self.set_status("Detecting text regions...")
        QtWidgets.QApplication.processEvents()

        self.detect_and_initialize_regions()

        self.setWindowTitle(f"Manga Helper GUI — {self.current_image_path.name}")
        self.set_status("Loaded image. Click a region in the list or on the preview.")

    # if text_regions is None use self.region_data instead
    def update_region_data(self, text_regions=None):
        if not text_regions and len(self.region_data) == 0:
            QtWidgets.QMessageBox.information(self, "No text found", "No text regions were detected in this image.")
            self.region_data = []
            self.selected_index = None
            self.refresh_ui()
            return
        
        if text_regions is None:
            sorted_regions = sort_text_regions([ (data["bbox"], data["ocr_text"], data.get("confidence")) for data in self.region_data ])
        else:
            sorted_regions = sort_text_regions(text_regions)
        
        # Decide whether to replace all (new image) or partial update
        replace_all = True
        update_index = -1
        for index, data in enumerate(self.region_data):
            if data["to_update"]:
                replace_all = False
                update_index = index

        # Reorder region_data to match sorted_regions indexing
        if not replace_all:
            # If partial update, we can try to match them by bbox
            new_region_data = []
            for index, sorted_region in enumerate(sorted_regions):
                matched = False
                for data in self.region_data:
                    if data["bbox"] == sorted_region[0]:
                        new_region_data.append(data)
                        matched = True
                        if data["to_update"]:
                            update_index = index
                        break
                if not matched:
                    # The only unmatched region should be the one we want to update
                    new_region_data.append(self.region_data[update_index])
                    update_index = index

            self.region_data = new_region_data
        
        # todo
        # Doesn't need to be done each time but for now it's faster since image is not loaded per instance but function call ...
        extracted_texts = extract_text(self.original_image, sorted_regions)
        translations = translate_with_context(extracted_texts)
        
        if replace_all:
            self.region_data = []
        for index, (region, text, translation) in enumerate(zip(sorted_regions, extracted_texts, translations)):
            if not replace_all and update_index != index:
                # Just update translation, other fields couldn't change
                data = self.region_data[index]
                data["translation"] = translation
                continue

            bbox = self.normalize_bbox(region[0])
            orientation = "vertical" if is_vertical_text(bbox) else "horizontal"
            data = {
                "bbox": bbox,
                "ocr_text": text,
                "hiragana": convert_to_hiragana(text),
                "romaji": convert_to_romaji(text),
                "translation": translation,
                "user_text": "",
                "orientation": orientation,
                "confidence": region[2] if len(region) > 2 else None,
                "to_update": False,
            }

            if not replace_all:
                self.region_data[index] = data
            else:
                self.region_data.append(data)

        if self.region_data:
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
            label = f"{index}. [{region['orientation']}] {region['ocr_text'][:20]}"
            self.region_list.addItem(label)

        self.selected_index = selected_index
        if self.selected_index is not None and 0 <= self.selected_index < len(self.region_data):
            self.region_list.setCurrentRow(self.selected_index)

    def update_region_details(self):
        if self.selected_index is None or self.selected_index >= len(self.region_data):
            self.detail_text.setPlainText("No text region selected.")
            self.user_text_edit.setText("")
            self.user_text_edit.setEnabled(False)
            return

        region = self.region_data[self.selected_index]
        lines = [
            "Japanese:",
            region['ocr_text'],
            "Hiragana:",
            region['hiragana'],
            "Romaji:",
            region['romaji'],
            "English:",
            region['translation'],
        ]
        self.detail_text.setPlainText("\n".join(lines))
        self.user_text_edit.setEnabled(True)
        self._ignore_user_text = True
        self.user_text_edit.setText(region['user_text'])
        self._ignore_user_text = False

    def enter_add_region_mode(self):
        if self.original_image is None:
            QtWidgets.QMessageBox.information(self, "Open Image", "Please open an image before adding a new region.")
            self.add_region_action.setChecked(False)
            return
        self.mode = "new_region"
        self.add_region_action.setChecked(True)
        self.update_mode_label()
        self.set_status("Drag on the preview to draw a new text region.")
        self.preview_widget.set_mode(self.mode)

    def enter_select_mode(self):
        self.mode = "select"
        self.add_region_action.setChecked(False)
        self.update_mode_label()
        self.set_status("Selection mode enabled.")
        self.preview_widget.set_mode(self.mode)

    def toggle_add_region_mode(self, checked):
        if checked:
            self.mode = "new_region"
            self.set_status("Drag on the preview to draw a new text region.")
        else:
            self.mode = "select"
            self.set_status("Selection mode enabled.")
        self.update_mode_label()
        self.preview_widget.set_mode(self.mode)

    def update_mode_label(self):
        self.mode_label.setText(f"Mode: {self.mode.replace('_', ' ').title()}")

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
        # bbox = region['bbox']
        # x0, y0 = region['bbox'][0]
        # x1, y1 = region['bbox'][2]
        # width = abs(x1 - x0)
        # # Textwrap to bbox width
        region['user_text'] = text
        self.unsaved_changes = True
        self.preview_widget.update_preview(self.original_image, self.region_data, self.selected_index, self)

    def on_font_settings_changed(self):
        self.font_family = self.font_entry.text()
        self.font_size = self.font_size_spin.value()
        self.text_center = self.center_check.isChecked()
        self.text_padding = self.padding_spin.value()
        self.preview_widget.update_preview(self.original_image, self.region_data, self.selected_index, self)

    def create_font(self):
        try:
            return ImageFont.truetype(self.font_family, max(8, self.font_size))
        except Exception:
            return ImageFont.load_default()

    def save_image(self):
        if self.original_image is None:
            QtWidgets.QMessageBox.information(self, "Save Image", "Open an image first before saving.")
            return

        dialog = QtWidgets.QFileDialog(self, "Save cleaned image")
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
        image_to_save = clean_image(self.original_image, [region['bbox'] for region in self.region_data], opacity=0.9)
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

    def export_text(self):
        if self.current_image_path is None:
            QtWidgets.QMessageBox.information(self, "Export Text", "Open an image first before exporting text.")
            return

        dialog = QtWidgets.QFileDialog(self, "Export detected text")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setNameFilters(["Text file (*.txt);;All files (*)"])
        dialog.setDefaultSuffix("txt")
        if dialog.exec():
            path = dialog.selectedFiles()[0]
        else:
            return

        if not path:
            return
        export_path = Path(path)

        with export_path.open("w", encoding="utf-8") as handle:
            handle.write(f"Image: {self.current_image_path.name}\n")
            handle.write(f"Regions: {len(self.region_data)}\n\n")
            for index, region in enumerate(self.region_data, start=1):
                handle.write(f"{index}. [{region['orientation']}]\n")
                handle.write(f"Japanese: {region['ocr_text']}\n")
                handle.write(f"Hiragana: {region['hiragana']}\n")
                handle.write(f"Romaji: {region['romaji']}\n")
                handle.write(f"English: {region['translation']}\n")
                if region['user_text'].strip():
                    handle.write(f"Replacement: {region['user_text']}\n")
                handle.write("\n")

        self.set_status(f"Text exported to {export_path.name}")

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
            event.accept()
        else:
            event.ignore()

    def render_user_text_overlay(self, image, text_bg_dilation=2):
        draw = ImageDraw.Draw(image)
        font = self.create_font()
        for region in self.region_data:
            text = region['user_text'].strip()
            if not text:
                continue

            x0, y0 = region['bbox'][0]
            x1, y1 = region['bbox'][2]
            padding = self.text_padding
            box = (x0 + padding, y0 + padding, x1 - padding, y1 - padding)
            text_size = draw.textbbox((0, 0), text, font=font)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]

            if self.text_center:
                px = box[0] + max(0, (box[2] - box[0] - text_width) // 2)
                py = box[1] + max(0, (box[3] - box[1] - text_height) // 2)
            else:
                px = box[0]
                py = box[1]

            # Draw text background - not a rectangle but the area of letters dilated by 2px
            # This is a more complex operation and requires getting the mask of the text and dilating it
            temp_image = Image.new("L", image.size, 0)
            temp_draw = ImageDraw.Draw(temp_image)
            temp_draw.text((px, py), text, fill=255, font=font)
            dilation_filter = ImageFilter.MaxFilter(1 + 2*text_bg_dilation)
            dilated_mask = temp_image.filter(dilation_filter)
            background_overlay = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image.paste(background_overlay, (0, 0), dilated_mask)

            draw.text((px, py), text, fill=(0, 0, 0, 255), font=font)

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
        # todo: fill values
        self.region_data.append({
            "bbox": bbox,
            "ocr_text": "",
            "hiragana": "",
            "romaji": "",
            "translation": "",
            "user_text": "",
            "orientation": "horizontal",
            "confidence": None,
            "to_update": True,
        })
        self.unsaved_changes = True
        self.selected_index = len(self.region_data) - 1
        self.update_region_data()
        self.refresh_ui()

    def update_region_bbox(self, index, bbox):
        """Update the bbox of a region."""
        if 0 <= index < len(self.region_data):
            self.region_data[index]['bbox'] = bbox
            self.unsaved_changes = True
            self.refresh_ui()


class PreviewWidget(QtWidgets.QLabel):
    EDGE_THRESHOLD = 10  # pixels for edge detection

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #222222;")
        self.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.current_image = None
        self.selected_index = None
        self.mode = "select"
        self.start_point = None
        self.preview_rect = None
        self.parent_window = parent
        self.dragging_edge = None
        self.hovered_edge = None

    def set_mode(self, mode):
        self.mode = mode

    def set_selected_index(self, index):
        self.selected_index = index

    def update_preview(self, image, region_data, selected_index, parent_window):
        self.parent_window = parent_window
        if image is None:
            self.clear()
            return

        preview = clean_image(image, [region["bbox"] for region in region_data], opacity=0.9)
        preview = parent_window.render_user_text_overlay(preview)
        if selected_index is not None and 0 <= selected_index < len(region_data):
            preview = self.render_selection_overlay(preview, region_data[selected_index]["bbox"])

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
                    bbox = region['bbox']
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
        
        bbox = self.parent_window.region_data[self.selected_index]['bbox']
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
            else:
                # Update cursor based on hovered edge
                img_coords = self.widget_to_image_coords(event.position())
                if img_coords:
                    edge = self.detect_edge(img_coords)
                    self.hovered_edge = edge
                    if edge:
                        self.setCursor(self.get_cursor_for_edge(edge))
                    else:
                        self.setCursor(QtCore.Qt.ArrowCursor)
                else:
                    self.setCursor(QtCore.Qt.ArrowCursor)

    def resize_bbox_by_edge(self, new_coords):
        """Resize the bbox by dragging an edge."""
        if self.selected_index is None or not self.parent_window:
            return

        # snapshot index
        index = self.selected_index
        
        region = self.parent_window.region_data[index]
        bbox = region['bbox']
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
        
        # Ensure minimum size and valid coordinates
        min_size = 20
        if abs(x1 - x0) >= min_size and abs(y1 - y0) >= min_size:
            new_bbox = [(min(x0, x1), min(y0, y1)), (max(x0, x1), min(y0, y1)), 
                       (max(x0, x1), max(y0, y1)), (min(x0, x1), max(y0, y1))]
            self.parent_window.update_region_bbox(index, new_bbox)

        self.selected_index = index

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
                return
            bbox = self.parent_window.region_bbox_from_points(img_start, img_end)
            for existing_region in self.parent_window.region_data:
                existing_bbox = existing_region['bbox']
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
                    return
                self.resize_bbox_by_edge(img_coords)

                # Trigger update to this area values and all areas' translations
                curr_region_data = self.parent_window.region_data[self.selected_index]
                curr_region_data["to_update"] = True
                self.parent_window.update_region_data()

            self.dragging_edge = None

    def widget_to_image_coords(self, point):
        if self.current_image is None:
            return None
        label_size = self.size()
        image_width, image_height = self.current_image.size
        if image_width == 0 or image_height == 0:
            return None

        scale = min(label_size.width() / image_width, label_size.height() / image_height)
        # px = (label_size.width() - image_width * scale) / 2
        # py = (label_size.height() - image_height * scale) / 2
        # this works
        px = py = 0
        x = (point.x() - px) / scale
        y = (point.y() - py) / scale
        if x < 0 or y < 0 or x > image_width or y > image_height:
            return None
        return (int(x), int(y))

    def render_selection_overlay(self, image, bbox):
        draw = ImageDraw.Draw(image)
        x0, y0 = bbox[0]
        x1, y1 = bbox[2]
        
        # Draw main rectangle
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0, 255), width=4)
        
        # Draw resize handles at corners and edges
        handle_size = 2
        handles = [
            ((x0, y0), "top-left"),
            ((x1, y0), "top-right"),
            ((x1, y1), "bottom-right"),
            ((x0, y1), "bottom-left"),
            (((x0 + x1) // 2, y0), "top"),
            (((x0 + x1) // 2, y1), "bottom"),
            ((x0, (y0 + y1) // 2), "left"),
            ((x1, (y0 + y1) // 2), "right"),
        ]
        
        for (hx, hy), edge_name in handles:
            # Draw handle rectangle
            draw.rectangle(
                [hx - handle_size, hy - handle_size, hx + handle_size, hy + handle_size],
                fill=(0, 255, 0, 200),
                outline=(0, 200, 0, 255),
                width=1
            )
        
        return image

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.preview_rect is not None:
            painter = QtGui.QPainter(self)
            painter.setPen(QtGui.QPen(QtGui.QColor("cyan"), 2, QtCore.Qt.DashLine))
            painter.drawRect(self.preview_rect)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MangaHelperGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
