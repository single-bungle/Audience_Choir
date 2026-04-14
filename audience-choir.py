import sys
import os
import tempfile
import soundfile as sf
import numpy as np
import time
import random
import concurrent.futures
from datetime import datetime
from argparse import Namespace
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QFileDialog, QLabel, QAbstractItemView,
    QFormLayout, QLineEdit, QSpinBox, QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QCheckBox, 
    QListWidget, QListWidgetItem, QSlider, QScrollArea, QProgressBar
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt, QPoint, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from pydub import AudioSegment
import librosa
from scipy.signal import fftconvolve
import pyloudnorm as pyln
import gc

# Add the directory containing the model code to the import path
sys.path.append(os.path.join(os.path.dirname(__file__), 'seed_vc'))

# Set BRIR directory
BRIR_DIR = os.path.join(os.path.dirname(__file__), 'D1-Brir')

# Global BRIR memory cache (removes disk I/O bottlenecks)
BRIR_CACHE = {}

def load_brir_cached(filepath):
    if filepath not in BRIR_CACHE:
        try:
            audio, sr = librosa.load(filepath, sr=None, mono=False)
            BRIR_CACHE[filepath] = (audio, sr)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load BRIR file: {filepath} ({str(e)})")
    return BRIR_CACHE[filepath]

def convolve_hrir(signal, hrir_L, hrir_R):
    left_output = fftconvolve(signal, hrir_L, mode='full')
    right_output = fftconvolve(signal, hrir_R, mode='full')
    output = np.vstack((left_output, right_output)).T
    return output

def spherical_to_cartesian(azimuth, elevation):
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    x = np.cos(el_rad) * np.cos(az_rad)
    y = np.cos(el_rad) * np.sin(az_rad)
    z = np.sin(el_rad)
    return np.array([x, y, z])

def slerp(v0, v1, t):
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if sin_theta < 1e-6:
        return (1 - t) * v0 + t * v1
    return (np.sin((1 - t) * theta) / sin_theta) * v0 + (np.sin(t * theta) / sin_theta) * v1

def interpolated_brir(brir_type, input_location):
    if brir_type == 'D1_brir':
        coordinates = [
            [198.4, -17.5], [251.6, -17.5], [225.0, 64.8], [0.0, 90.0], [341.6, 17.5], 
            [288.4, -17.5], [45.0, 35.3], [315.0, -64.8], [270.0, 45.0], [251.6, 17.5], 
            [288.4, 17.5], [71.6, -17.5], [90.0, 45.0], [90.0, -45.0], [161.6, 17.5], 
            [0.0, -45.0], [315.0, -35.3], [45.0, 64.8], [341.6, -17.5], [0.0, 0.0], 
            [90.0, 0.0], [270.0, 0.0], [225.0, 35.3], [135.0, 0.0], [18.4, -17.5], 
            [18.4, 17.5], [135.0, -35.3], [108.4, -17.5], [198.4, 17.5], [315.0, 35.3], 
            [45.0, -64.8], [0.0, -90.0], [225.0, -35.3], [180.0, -45.0], [135.0, 64.8], 
            [161.6, -17.5], [135.0, 35.3], [315.0, 0.0], [108.4, 17.5], [225.0, -64.8], 
            [180.0, 0.0], [45.0, -35.3], [45.0, 0.0], [0.0, 45.0], [225.0, 0.0], 
            [315.0, 64.8], [71.6, 17.5], [270.0, -45.0], [180.0, 45.0], [135.0, -64.8]
        ]

    cartesian_coords = [spherical_to_cartesian(az, el) for az, el in coordinates]
    desired_azimuth = input_location[0]
    desired_elevation = input_location[1]
    desired_cartesian = spherical_to_cartesian(desired_azimuth, desired_elevation)

    distances = [np.linalg.norm(desired_cartesian - c) for c in cartesian_coords]
    closest_indices = np.argsort(distances)[:2]
    hrir_1_loc = coordinates[closest_indices[0]]
    hrir_2_loc = coordinates[closest_indices[1]]

    hrir_1_path = os.path.join(BRIR_DIR, f"azi_{hrir_1_loc[0]}_ele_{hrir_1_loc[1]}.wav")
    hrir_2_path = os.path.join(BRIR_DIR, f"azi_{hrir_2_loc[0]}_ele_{hrir_2_loc[1]}.wav")

    hrir_1, sr_hrir_1 = load_brir_cached(hrir_1_path)
    hrir_2, sr_hrir_2 = load_brir_cached(hrir_2_path)

    if sr_hrir_1 != sr_hrir_2:
        raise ValueError("Sample rates of BRIR files do not match.")

    hrir_1_L, hrir_1_R = hrir_1[0], hrir_1[1]
    hrir_2_L, hrir_2_R = hrir_2[0], hrir_2[1]

    t = distances[closest_indices[0]] / (distances[closest_indices[0]] + distances[closest_indices[1]])
    interpolated_hrir_L = (1 - t) * hrir_1_L + t * hrir_2_L
    interpolated_hrir_R = (1 - t) * hrir_1_R + t * hrir_2_R
    
    return interpolated_hrir_L, interpolated_hrir_R, sr_hrir_1

def apply_brir_convolution_task(file_path, lon, lat, target_filename):
    try:
        audio_data, sr = librosa.load(file_path, sr=None, mono=True)
        hrir_L, hrir_R, sr_hrir = interpolated_brir('D1_brir', [-lon, lat])
        
        if sr != sr_hrir:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sr_hrir)
            sr = sr_hrir
            
        convolved_signal = convolve_hrir(audio_data, hrir_L, hrir_R)
        convolved_file = os.path.join(tempfile.gettempdir(), f"convolved_{target_filename}")
        sf.write(convolved_file, convolved_signal, sr)
        
        convolved_audio = AudioSegment.from_wav(convolved_file)
        return (convolved_file, target_filename, convolved_audio)
    except Exception as e:
        raise RuntimeError(f"Convolution failed for {target_filename}: {str(e)}")


class ConversionWorker(QThread):
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    finished = pyqtSignal(list, float)
    error = pyqtSignal(str)

    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.args = args

    def run(self):
        from seed_vc import inference
        import glob
        from pydub import AudioSegment

        start_time = time.time()
        converted_files = []
        total_targets = len(self.args.targets)

        # Setup to intercept terminal output
        interceptor = OutputInterceptor()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = interceptor
        sys.stderr = interceptor  # Progress bars like tqdm usually output to stderr

        try:
            os.makedirs(self.args.output, exist_ok=True)
            for idx, t_path in enumerate(self.args.targets):
                self.args.target = t_path
                target_filename = os.path.basename(t_path)
                self.progress.emit(f"Converting {target_filename}...")
                
                # Calculate which section of the total progress the current file occupies
                base_progress = (idx / total_targets) * 100
                progress_chunk = 100 / total_targets

                # Internal function to extract percentage (%) from console output and update UI
                def parse_progress(text):
                    # Format 1: Find percentage format like "45%"
                    match_percent = re.search(r'(\d{1,3})%', text)
                    if match_percent:
                        percent = int(match_percent.group(1))
                        current_progress = int(base_progress + (percent / 100.0) * progress_chunk)
                        self.progress_value.emit(current_progress)
                        return

                    # Format 2: Find step format like "25/50"
                    match_step = re.search(r'(\d+)/(\d+)', text)
                    if match_step:
                        current_step = int(match_step.group(1))
                        total_step = int(match_step.group(2))
                        if total_step > 0:
                            percent = (current_step / total_step) * 100
                            current_progress = int(base_progress + (percent / 100.0) * progress_chunk)
                            self.progress_value.emit(current_progress)

                # Connect signals
                interceptor.output_signal.connect(parse_progress)
                
                try:
                    inference(self.args) # The function above captures logs output inside here
                    
                    # Disconnect signal (prevent duplication for next target conversion)
                    interceptor.output_signal.disconnect(parse_progress)

                    output_files = glob.glob(os.path.join(self.args.output, "*.wav"))
                    if not output_files:
                        self.progress.emit(f"No output file for {target_filename}.")
                        continue
                        
                    latest_file = max(output_files, key=os.path.getmtime)
                    audio = AudioSegment.from_wav(latest_file)
                    normalized_audio = self.parent().normalize_audio_power(audio)
                    normalized_file = os.path.join(tempfile.gettempdir(), f"normalized_{target_filename}")
                    normalized_audio.export(normalized_file, format="wav")
                    
                    converted_files.append((normalized_file, target_filename, normalized_audio))
                    
                    # Upon target completion, ensure it reaches exactly 100% of its section
                    self.progress_value.emit(int(base_progress + progress_chunk))
                    
                except Exception as e:
                    self.progress.emit(f"Conversion failed for {target_filename}: {str(e)}")
                    continue
                    
            elapsed_time = time.time() - start_time
            if not converted_files:
                self.progress.emit(f"No converted audio files. (Time elapsed: {elapsed_time:.2f}s)")
            else:
                self.progress.emit(f"Conversion complete. Files: {len(converted_files)} (Time elapsed: {elapsed_time:.2f}s)")
            
            self.progress_value.emit(100) # Final 100% confirmed
            self.finished.emit(converted_files, elapsed_time)

        except Exception as e:
            elapsed_time = time.time() - start_time
            self.error.emit(f"General error: {str(e)} (Time elapsed: {elapsed_time:.2f}s)")
            
        finally:
            # Restore original console output when conversion is completely done (very important!)
            sys.stdout = original_stdout
            sys.stderr = original_stderr

class SpatialAudioWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list, float)
    error = pyqtSignal(str)

    def __init__(self, converted_files, circle_positions, parent=None):
        super().__init__(parent)
        self.converted_files = converted_files
        self.circle_positions = circle_positions

    def run(self):
        start_time = time.time()
        new_converted_files = []
        tasks = []

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                for normalized_file, target_filename, _ in self.converted_files:
                    name_without_ext = os.path.splitext(target_filename)[0]
                    if name_without_ext not in self.circle_positions:
                        self.progress.emit(f"No coordinates for {target_filename}.")
                        continue

                    lon, lat = self.circle_positions[name_without_ext]
                    self.progress.emit(f"Scheduling parallel spatial audio processing for {target_filename}...")
                    
                    future = executor.submit(apply_brir_convolution_task, normalized_file, lon, lat, target_filename)
                    tasks.append(future)

                for future in concurrent.futures.as_completed(tasks):
                    result = future.result()
                    new_converted_files.append(result)
                    self.progress.emit(f"Spatial audio rendering complete for {result[1]}.")

            elapsed_time = time.time() - start_time

            if not new_converted_files:
                self.progress.emit(f"No spatial audio applied files. (Time elapsed: {elapsed_time:.2f}s)")
                self.finished.emit([], elapsed_time)
            else:
                self.finished.emit(new_converted_files, elapsed_time)

        except Exception as e:
            elapsed_time = time.time() - start_time
            self.error.emit(f"Spatial audio application error: {str(e)} (Time elapsed: {elapsed_time:.2f}s)")
        finally:
            gc.collect()


class TargetAudioControl(QWidget):
    def __init__(self, title, visualizer):
        super().__init__()
        self.media_player = QMediaPlayer()
        self.audio_files = []
        self.audio_data_dict = {}
        self.visualizer = visualizer

        self.upload_button = QPushButton("Upload WAV Files")
        self.play_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        self.list_widget = QListWidget()
        self.label = QLabel("No files uploaded.")

        self.upload_button.clicked.connect(self.upload_audio)
        self.play_button.clicked.connect(self.play_audio)
        self.stop_button.clicked.connect(self.stop_audio)
        self.list_widget.itemChanged.connect(self.toggle_circle_from_item)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        layout.addWidget(self.upload_button)
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.current_selected_path = None

    def upload_audio(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select WAV Files", "", "WAV Files (*.wav)")
        if not files:
            return
        for file_path in files:
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            if file_path not in self.audio_files:
                self.audio_files.append(file_path)
                item = QListWidgetItem(filename)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.list_widget.addItem(item)
                try:
                    audio_data = AudioSegment.from_wav(file_path)
                    self.audio_data_dict[name_without_ext] = audio_data
                except Exception as e:
                    self.label.setText(f"Error loading {filename}: {str(e)}")
        self.label.setText(f"{len(self.audio_files)} files uploaded.")

    def toggle_circle_from_item(self, item):
        filename = item.text()
        name_without_ext = os.path.splitext(filename)[0]
        if item.checkState() == Qt.Checked:
            self.visualizer.add_circle(name_without_ext)
        else:
            if name_without_ext in self.visualizer.circles:
                del self.visualizer.circles[name_without_ext]
                self.visualizer.update()

    def play_audio(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return
        selected_name = selected_items[0].text()
        for path in self.audio_files:
            if os.path.basename(path) == selected_name:
                self.current_selected_path = path
                self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
                break
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")
        else:
            self.media_player.play()
            self.play_button.setText("Pause")

    def stop_audio(self):
        self.media_player.stop()
        self.play_button.setText("Play")


class SourceAudioControl(QWidget):
    def __init__(self, title):
        super().__init__()
        self.media_player = QMediaPlayer()
        self.audio_path = None
        self.audio_data = None

        self.upload_button = QPushButton("Upload WAV File")
        self.play_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        self.label = QLabel("No file uploaded.")

        self.upload_button.clicked.connect(self.upload_audio)
        self.play_button.clicked.connect(self.play_audio)
        self.stop_button.clicked.connect(self.stop_audio)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        layout.addWidget(self.upload_button)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def upload_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a WAV File", "", "WAV Files (*.wav)")
        if not file_path:
            return
        try:
            self.audio_path = file_path
            self.audio_data = AudioSegment.from_wav(file_path)
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            self.label.setText(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            self.label.setText(f"Error loading file: {str(e)}")

    def play_audio(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")
        else:
            self.media_player.play()
            self.play_button.setText("Pause")

    def stop_audio(self):
        self.media_player.stop()
        self.play_button.setText("Play")

    @property
    def current_selected_path(self):
        return self.audio_path


class DraggableTargetCircle(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 200)
        self.setStyleSheet("background-color: white;")
        self.circles = {}
        self.dragging_circle = None
        self.drag_offset = QPoint(0, 0)
        self.logic_x_range = (-180, 180)
        self.logic_y_range = (-90, 90)

    def add_circle(self, name):
        center = QPoint(self.width() // 2, self.height() // 2)
        self.circles[name] = center
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        origin_x, origin_y = w // 2, h // 2
        scale_x = w / (self.logic_x_range[1] - self.logic_x_range[0])
        scale_y = h / (self.logic_y_range[1] - self.logic_y_range[0])
        grid_spacing_deg = 30
        painter.setPen(QPen(QColor(230, 230, 230), 1))

        for lon in range(self.logic_x_range[0], self.logic_x_range[1] + 1, grid_spacing_deg):
            x = origin_x + lon * scale_x
            painter.drawLine(int(x), 0, int(x), h)
            painter.setFont(QFont("Arial", 8))
            painter.setPen(Qt.darkGray)
            painter.drawText(int(x), h - 15, 20, 20, Qt.AlignCenter, str(lon))

        for lat in range(self.logic_y_range[0], self.logic_y_range[1], grid_spacing_deg):
            y = origin_y - lat * scale_y
            painter.drawLine(0, int(y), w, int(y))
            painter.setFont(QFont("Arial", 8))
            painter.setPen(Qt.darkGray)
            painter.drawText(5, int(y) - 10, 30, 20, Qt.AlignLeft, str(lat))

        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(origin_x, 0, origin_x, h)
        painter.drawLine(0, origin_y, w, origin_y)

        for name, pos in self.circles.items():
            painter.setBrush(QColor(200, 200, 255))
            painter.setPen(QPen(Qt.blue, 2))
            painter.drawEllipse(pos, 15, 15)
            painter.setFont(QFont("Arial", 8))
            painter.setPen(Qt.black)
            painter.drawText(painter.boundingRect(pos.x() - 30, pos.y() - 10, 60, 20, Qt.AlignCenter, name), Qt.AlignCenter, name)

    def mousePressEvent(self, event):
        for name, pos in self.circles.items():
            if (pos - event.pos()).manhattanLength() < 30:
                self.dragging_circle = name
                self.drag_offset = event.pos() - pos
                break

    def mouseMoveEvent(self, event):
        if self.dragging_circle:
            new_pos = event.pos() - self.drag_offset
            w, h = self.width(), self.height()
            origin_x, origin_y = w // 2, h // 2
            scale_x = w / (self.logic_x_range[1] - self.logic_x_range[0])
            scale_y = h / (self.logic_y_range[1] - self.logic_y_range[0])
            logic_x = (new_pos.x() - origin_x) / scale_x
            logic_y = (origin_y - new_pos.y()) / scale_y
            logic_x = max(self.logic_x_range[0], min(self.logic_x_range[1], logic_x))
            logic_y = max(self.logic_y_range[0], min(self.logic_y_range[1], logic_y))
            clipped_x = origin_x + logic_x * scale_x
            clipped_y = origin_y - logic_y * scale_y
            self.circles[self.dragging_circle] = QPoint(int(clipped_x), int(clipped_y))
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_circle = None

    def get_circle_positions(self):
        w, h = self.width(), self.height()
        origin_x, origin_y = w // 2, h // 2
        scale_x = (self.logic_x_range[1] - self.logic_x_range[0]) / w
        scale_y = (self.logic_y_range[1] - self.logic_y_range[0]) / h
        positions = {}
        for name, pos in self.circles.items():
            lon = (pos.x() - origin_x) * scale_x
            lat = (origin_y - pos.y()) * scale_y
            positions[name] = (lon, lat)
        return positions


class ChorusAudioControl(QWidget):
    def __init__(self, title):
        super().__init__()
        self.media_player = QMediaPlayer()
        self.audio_path = None

        self.play_button = QPushButton("Play Chorus")
        self.stop_button = QPushButton("Stop Chorus")
        self.volume_label = QLabel("Chorus Volume: 1.00")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 200)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self.update_volume)
        self.label = QLabel("No chorus audio available.")

        self.play_button.clicked.connect(self.play_audio)
        self.stop_button.clicked.connect(self.stop_audio)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(self.volume_label)
        volume_layout.addWidget(self.volume_slider)

        layout.addLayout(button_layout)
        layout.addLayout(volume_layout)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def add_chorus_audio(self, file_path, audio_segment):
        self.audio_path = file_path
        self.label.setText(f"Loaded: {os.path.basename(file_path)}")
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))

    def update_volume(self):
        volume = self.volume_slider.value() / 100.0
        self.volume_label.setText(f"Chorus Volume: {volume:.2f}")
        self.media_player.setVolume(int(volume * 100))

    def play_audio(self):
        if not self.audio_path: return
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play Chorus")
        else:
            self.media_player.play()
            self.play_button.setText("Pause Chorus")

    def stop_audio(self):
        self.media_player.stop()
        self.play_button.setText("Play Chorus")


class SpatialChorusControl(QWidget):
    def __init__(self, title):
        super().__init__()
        self.media_player = QMediaPlayer()
        self.audio_path = None

        self.play_button = QPushButton("Play Spatial Chorus")
        self.stop_button = QPushButton("Stop Spatial Chorus")
        self.volume_label = QLabel("Spatial Chorus Volume: 1.00")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 200)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self.update_volume)
        self.label = QLabel("No spatial chorus audio available.")

        self.play_button.clicked.connect(self.play_audio)
        self.stop_button.clicked.connect(self.stop_audio)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"<b>{title}</b>"))
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(self.volume_label)
        volume_layout.addWidget(self.volume_slider)

        layout.addLayout(button_layout)
        layout.addLayout(volume_layout)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def add_spatial_chorus_audio(self, file_path, audio_segment):
        self.audio_path = file_path
        self.label.setText(f"Loaded: {os.path.basename(file_path)}")
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))

    def update_volume(self):
        volume = self.volume_slider.value() / 100.0
        self.volume_label.setText(f"Spatial Chorus Volume: {volume:.2f}")
        self.media_player.setVolume(int(volume * 100))

    def play_audio(self):
        if not self.audio_path: return
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play Spatial Chorus")
        else:
            self.media_player.play()
            self.play_button.setText("Pause Spatial Chorus")

    def stop_audio(self):
        self.media_player.stop()
        self.play_button.setText("Play Spatial Chorus")


class AudioPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Conversion & Spatial Choir Tool")
        self.setGeometry(100, 100, 1200, 800)

        # ==========================================
        # 1. Custom Widget and Base UI Controller Initialization
        # ==========================================
        self.visualizer = DraggableTargetCircle()
        self.audio_target = TargetAudioControl("Singer Feature", self.visualizer)
        self.audio_source = SourceAudioControl("Song")
        self.chorus_control = ChorusAudioControl("Chorus (Non-Spatial)")
        self.spatial_chorus_control = SpatialChorusControl("Spatial Chorus")

        self.converted_player = QMediaPlayer()
        self.converted_files = []
        self.original_converted_files = []
        self.config_form = self.create_config_form()

        # Converted audio list
        self.converted_audio_list = QListWidget()
        self.converted_audio_list.itemClicked.connect(self.play_selected_converted_audio)

        # Conversion related UI
        self.convert_button = QPushButton("Convert")
        self.convert_button.clicked.connect(self.start_conversion_thread)
        self.estimated_time_label = QLabel("Estimated Inference Time: N/A")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Conversion Progress: %p%")

        # Spatial audio related UI
        self.spatial_button = QPushButton("Apply Spatial Audio")
        self.spatial_button.clicked.connect(self.start_spatial_audio_thread)
        self.spatial_button.setEnabled(False)

        # Save related buttons
        self.save_button = QPushButton("Save Converted Audio")
        self.save_button.clicked.connect(self.save_audio)
        self.save_button.setEnabled(False)

        self.save_spatial_button = QPushButton("Save Spatial Audio")
        self.save_spatial_button.clicked.connect(self.save_spatial_audio)
        self.save_spatial_button.setEnabled(False)

        self.result_label = QLabel("No output yet.")

        # ==========================================
        # 2. Final Master Processing New UI 
        # ==========================================
        self.global_reverb_toggle = QCheckBox("Enable Global Reverb")
        
        self.reverb_intensity_label = QLabel("Reverb Intensity: 15%")
        self.reverb_intensity_slider = QSlider(Qt.Horizontal)
        self.reverb_intensity_slider.setRange(0, 100)
        self.reverb_intensity_slider.setValue(15)
        self.reverb_intensity_slider.valueChanged.connect(self.update_reverb_label)

        self.apply_button = QPushButton("Apply All Settings & Mix Final")
        self.apply_button.clicked.connect(self.generate_final_output)
        
        self.final_play_button = QPushButton("Play Final Output")
        self.final_stop_button = QPushButton("Stop Final Output")
        self.final_label = QLabel("No final output generated.")
        self.final_player = QMediaPlayer()
        
        self.save_final_button = QPushButton("Save Final Output")
        self.save_final_button.clicked.connect(self.save_final_audio)
        self.save_final_button.setEnabled(False)

        self.final_play_button.clicked.connect(self.play_final_audio)
        self.final_stop_button.clicked.connect(self.stop_final_audio)

        # ==========================================
        # 3. Overall Layout Setup
        # ==========================================
        main_layout = QHBoxLayout()
        
        # --- Left panel setup ---
        left_panel = QScrollArea()
        left_panel.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        left_layout.addWidget(self.audio_target)
        left_layout.addWidget(self.audio_source)
        left_layout.addWidget(QLabel("<b>Conversion Settings</b>"))
        left_layout.addLayout(self.config_form)
        left_layout.addWidget(self.convert_button)
        left_layout.addWidget(self.estimated_time_label)
        left_layout.addWidget(self.progress_bar)
        
        left_layout.addWidget(QLabel("<b>Converted Audio Results</b>"))
        left_layout.addWidget(self.converted_audio_list) 
        left_layout.addWidget(self.chorus_control)
        left_layout.addWidget(self.save_button) 
        
        left_layout.addStretch()
        left_widget.setLayout(left_layout)
        left_panel.setWidget(left_widget)

        # --- Right panel setup ---
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        right_layout.addWidget(QLabel("<b>Spatial Positioning</b>"))
        right_layout.addWidget(self.visualizer)
        right_layout.addWidget(self.spatial_button)
        right_layout.addWidget(self.spatial_chorus_control)
        right_layout.addWidget(self.save_spatial_button)
        right_layout.addWidget(self.result_label)
        
        # Final Master Processing layout
        right_layout.addWidget(QLabel("<b>Final Master Processing</b>"))
        right_layout.addWidget(self.global_reverb_toggle)
        
        reverb_layout = QHBoxLayout()
        reverb_layout.addWidget(self.reverb_intensity_label)
        reverb_layout.addWidget(self.reverb_intensity_slider)
        right_layout.addLayout(reverb_layout)
        
        right_layout.addWidget(self.apply_button)
        
        final_btn_layout = QHBoxLayout()
        final_btn_layout.addWidget(self.final_play_button)
        final_btn_layout.addWidget(self.final_stop_button)
        right_layout.addLayout(final_btn_layout)
        
        right_layout.addWidget(self.final_label)
        
        # Place Save Final Output button at the very bottom of the right panel
        right_layout.addWidget(self.save_final_button)
        
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        right_panel.setWidget(right_widget)

        # --- Merge into main window ---
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # ==========================================
        # 4. Signal Connections (for Estimated Time updates)
        # ==========================================
        self.audio_source.upload_button.clicked.connect(self.update_estimated_time)
        self.audio_target.upload_button.clicked.connect(self.update_estimated_time)
        self.audio_target.list_widget.itemChanged.connect(self.update_estimated_time)
        
        # Add parameter integration signals if available
        self.diffusion_steps_input.valueChanged.connect(self.update_estimated_time)
        self.checkpoint_input.textChanged.connect(self.update_estimated_time)

    def update_estimated_time(self):
        pass 

    def create_config_form(self):
        self.output_input = QLineEdit("./reconstructed")
        self.diffusion_steps_input = QSpinBox()
        self.diffusion_steps_input.setValue(50)
        self.length_adjust_input = QDoubleSpinBox()
        self.length_adjust_input.setValue(1.0)
        self.inference_cfg_input = QDoubleSpinBox()
        self.inference_cfg_input.setValue(1.0)
        self.f0_condition_input = QCheckBox()
        self.f0_condition_input.setChecked(True)
        self.auto_f0_adjust_input = QCheckBox()
        self.auto_f0_adjust_input.setChecked(False)
        self.fp16_input = QCheckBox()
        self.fp16_input.setChecked(True)
        self.semitone_shift_input = QSpinBox()
        self.semitone_shift_input.setRange(-24, 24)
        self.semitone_shift_input.setValue(0)
        self.checkpoint_input = QLineEdit()
        self.config_input = QLineEdit()

        main_form_layout = QHBoxLayout()
        left_form = QFormLayout()
        right_form = QFormLayout()

        left_form.addRow("Output Dir", self.output_input)
        left_form.addRow("Diffusion Steps", self.diffusion_steps_input)
        left_form.addRow("Length Adjust", self.length_adjust_input)
        left_form.addRow("Inference CFG Rate", self.inference_cfg_input)
        left_form.addRow("Use F0 Condition", self.f0_condition_input)

        right_form.addRow("Auto F0 Adjust", self.auto_f0_adjust_input)
        right_form.addRow("Semitone Shift", self.semitone_shift_input)
        right_form.addRow("Checkpoint Path", self.checkpoint_input)
        right_form.addRow("Config Path", self.config_input)
        right_form.addRow("Use FP16", self.fp16_input)

        main_form_layout.addLayout(left_form)
        main_form_layout.addLayout(right_form)
        
        return main_form_layout

    def normalize_audio_power(self, audio):
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        sr = audio.frame_rate

        samples = samples.astype(np.float32) / 32768.0
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(samples)
        target_loudness = -23.0
        normalized_samples = pyln.normalize.loudness(samples, loudness, target_loudness)
        normalized_samples = (normalized_samples * 32768).astype(np.int16)
        normalized_audio = AudioSegment(
            normalized_samples.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )
        return normalized_audio

    def update_non_spatial_chorus(self):
        if not self.original_converted_files:
            self.chorus_control.label.setText("No chorus audio available.")
            return

        try:
            sr = self.original_converted_files[0][2].frame_rate
            max_len = max(len(audio) for _, _, audio in self.original_converted_files)
            max_samples = int((max_len / 1000.0 + 0.05) * sr) 
            combined_samples = np.zeros(max_samples, dtype=np.float32)
            
            num_audios = len(self.original_converted_files)
            headroom_db = -6.0

            for file_path, target_filename, audio in self.original_converted_files:
                volume = 1.0 # Individual volume removed
                samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)

                delay_ms = random.uniform(10.0, 35.0)
                delay_samples = int((delay_ms / 1000.0) * sr)
                
                delayed_samples = np.pad(samples, (delay_samples, 0), mode='constant')
                
                if len(delayed_samples) < len(combined_samples):
                    delayed_samples = np.pad(delayed_samples, (0, len(combined_samples) - len(delayed_samples)), mode='constant')
                elif len(delayed_samples) > len(combined_samples):
                    delayed_samples = delayed_samples[:len(combined_samples)]
                    
                combined_samples += delayed_samples * volume * 0.5 / num_audios

            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(combined_samples)
            normalized_samples = pyln.normalize.loudness(combined_samples, loudness, -23.0)
            normalized_samples = (normalized_samples * 32768).astype(np.int16)

            combined_audio = AudioSegment(
                normalized_samples.tobytes(), frame_rate=sr, sample_width=2, channels=1
            )

            if combined_audio.max_dBFS > headroom_db:
                gain_adjust = headroom_db - combined_audio.max_dBFS
                combined_audio = combined_audio.apply_gain(gain_adjust)

            combined_file = os.path.join(tempfile.gettempdir(), "non_spatial_chorus.wav")
            combined_audio.export(combined_file, format="wav")
            self.chorus_control.add_chorus_audio(combined_file, combined_audio)
            self.result_label.setText("Non-spatial chorus updated successfully with Random Delay.")
        except Exception as e:
            self.result_label.setText(f"Non-spatial chorus generation failed: {str(e)}")

    def update_spatial_chorus(self):
        if not self.converted_files:
            self.spatial_chorus_control.label.setText("No spatial chorus audio available.")
            return

        try:
            sr = self.converted_files[0][2].frame_rate
            circle_positions = self.visualizer.get_circle_positions()
            
            # Filter only valid targets with existing coordinates
            valid_files = [f for f in self.converted_files if os.path.splitext(f[1])[0] in circle_positions]
            num_audios = len(valid_files)
            
            if num_audios == 0:
                self.result_label.setText("No target audio with coordinates set.")
                return
            
            max_len = max(len(audio) for _, _, audio in valid_files)
            max_samples = int((max_len / 1000.0 + 0.05) * sr)
            combined_samples = np.zeros((max_samples, 2), dtype=np.float32)
            headroom_db = -6.0

            for file_path, target_filename, audio in valid_files:
                volume = 1.0 # Individual volume removed
                audio_data, file_sr = librosa.load(file_path, sr=None, mono=False)
                
                # audio_data is (2, N) shape, so apply .T to match (N, 2)
                samples = audio_data.T if audio_data.ndim == 2 else np.vstack((audio_data, audio_data)).T
                
                delay_ms = random.uniform(10.0, 35.0)
                delay_samples = int((delay_ms / 1000.0) * sr)
                
                delayed_samples = np.pad(samples, ((delay_samples, 0), (0, 0)), mode='constant')

                if delayed_samples.shape[0] < max_samples:
                    delayed_samples = np.pad(delayed_samples, ((0, max_samples - delayed_samples.shape[0]), (0, 0)), mode='constant')
                elif delayed_samples.shape[0] > max_samples:
                    delayed_samples = delayed_samples[:max_samples, :]

                combined_samples += delayed_samples * volume * 0.5 / num_audios

            # Modified core part: removed .T to maintain (N, 2) dimension required by pyloudnorm
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(combined_samples)
            normalized_samples = pyln.normalize.loudness(combined_samples, loudness, -23.0)
            normalized_samples = (normalized_samples * 32768).astype(np.int16)

            combined_audio = AudioSegment(
                normalized_samples.tobytes(), frame_rate=sr, sample_width=2, channels=2
            )

            if combined_audio.max_dBFS > headroom_db:
                gain_adjust = headroom_db - combined_audio.max_dBFS
                combined_audio = combined_audio.apply_gain(gain_adjust)

            # Assign timestamp to temp filename to prevent player lock errors
            temp_filename = f"spatial_chorus_{int(time.time())}.wav"
            combined_file = os.path.join(tempfile.gettempdir(), temp_filename)
            combined_audio.export(combined_file, format="wav")
            
            self.spatial_chorus_control.add_spatial_chorus_audio(combined_file, combined_audio)
            self.result_label.setText("Spatial chorus updated successfully with Random Delay.")
            
        except Exception as e:
            self.result_label.setText(f"Spatial chorus generation failed: {str(e)}")

    def get_args(self):
        source = self.audio_source.current_selected_path
        if not source or not os.path.isfile(source):
            raise ValueError("Source audio file is not selected or invalid.")

        checked_targets = []
        for i in range(self.audio_target.list_widget.count()):
            item = self.audio_target.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                filename = item.text()
                target_path = next((path for path in self.audio_target.audio_files if os.path.basename(path) == filename), None)
                if target_path and os.path.isfile(target_path):
                    checked_targets.append(target_path)

        if not checked_targets:
            raise ValueError("There is no checked Singer Feature audio file. Please check at least one target audio.")

        return Namespace(
            source=source,
            targets=checked_targets,
            output=self.output_input.text(),
            diffusion_steps=self.diffusion_steps_input.value(),
            length_adjust=self.length_adjust_input.value(),
            inference_cfg_rate=self.inference_cfg_input.value(),
            f0_condition=self.f0_condition_input.isChecked(),
            auto_f0_adjust=self.auto_f0_adjust_input.isChecked(),
            semi_tone_shift=self.semitone_shift_input.value(),
            checkpoint=self.checkpoint_input.text() or None,
            config=self.config_input.text() or None,
            fp16=self.fp16_input.isChecked()
        )

    def start_conversion_thread(self):
        try:
            args = self.get_args()
            self.convert_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.result_label.setText("Converting... please wait.")
            self.conversion_worker = ConversionWorker(args, self)
            self.conversion_worker.progress.connect(self.result_label.setText)
            self.conversion_worker.progress_value.connect(self.progress_bar.setValue)
            self.conversion_worker.finished.connect(self.on_conversion_finished)
            self.conversion_worker.error.connect(self.on_conversion_error)
            self.conversion_worker.start()
        except ValueError as e:
            self.result_label.setText(str(e))

    def on_conversion_finished(self, converted_files, elapsed_time):
        self.converted_files = converted_files
        self.original_converted_files = converted_files[:]
        self.converted_audio_list.clear()

        # Volume slider UI removed, adding simple list
        for normalized_file, target_filename, normalized_audio in converted_files:
            item = QListWidgetItem(f"Converted: {target_filename}")
            item.setData(Qt.UserRole, normalized_file)
            self.converted_audio_list.addItem(item)

        self.convert_button.setEnabled(True)
        self.progress_bar.setValue(100)
        if converted_files:
            first_file, first_target, _ = converted_files[0]
            self.converted_player.setMedia(QMediaContent(QUrl.fromLocalFile(first_file)))
            self.converted_player.play()
            self.result_label.setText(f"Playing: Converted {first_target} (Time elapsed: {elapsed_time:.2f}s)")
            self.spatial_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.update_non_spatial_chorus()
        else:
            self.result_label.setText(f"No converted audio files. (Time elapsed: {elapsed_time:.2f}s)")
            self.spatial_button.setEnabled(False)
            self.save_button.setEnabled(False)

        self.save_spatial_button.setEnabled(False)

    def on_conversion_error(self, error_message):
        self.convert_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.result_label.setText(error_message)
        self.spatial_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.save_spatial_button.setEnabled(False)

    def start_spatial_audio_thread(self):
        circle_positions = self.visualizer.get_circle_positions()
        self.spatial_button.setEnabled(False)
        self.result_label.setText("Applying spatial audio... please wait.")
        self.spatial_worker = SpatialAudioWorker(self.original_converted_files, circle_positions, self)
        self.spatial_worker.progress.connect(self.result_label.setText)
        self.spatial_worker.finished.connect(self.on_spatial_audio_finished)
        self.spatial_worker.error.connect(self.on_spatial_audio_error)
        self.spatial_worker.start()

    def on_spatial_audio_finished(self, new_converted_files, elapsed_time):
        self.converted_files = new_converted_files
        self.converted_audio_list.clear()

        # Volume slider UI removed
        for convolved_file, target_filename, convolved_audio in new_converted_files:
            pos_info = self.visualizer.get_circle_positions()[os.path.splitext(target_filename)[0]]
            item = QListWidgetItem(f"Converted: {target_filename} (lon: {pos_info[0]:.2f}, lat: {pos_info[1]:.2f})")
            item.setData(Qt.UserRole, convolved_file)
            self.converted_audio_list.addItem(item)

        self.spatial_button.setEnabled(True)
        self.save_spatial_button.setEnabled(bool(new_converted_files))
        self.update_spatial_chorus()
        self.result_label.setText(f"Spatial audio applied (Time elapsed: {elapsed_time:.2f}s)")

    def on_spatial_audio_error(self, error_message):
        self.spatial_button.setEnabled(True)
        self.save_spatial_button.setEnabled(False)
        self.result_label.setText(error_message)

    def play_selected_converted_audio(self, item):
        file_path = item.data(Qt.UserRole)
        if file_path and os.path.isfile(file_path):
            self.converted_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            self.converted_player.play()
            self.result_label.setText(f"Playing: {item.text()}")

    # --- Newly Added: Self-generated Reverb Effect (Using scipy.signal) ---
    def apply_reverb_effect(self, audio_segment):
        sr = audio_segment.frame_rate
        channels = audio_segment.channels
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        
        # Generate 1.5s impulse response (White Noise * Exponential Decay)
        reverb_time = 1.5
        t = np.linspace(0, reverb_time, int(sr * reverb_time), endpoint=False)
        ir = np.random.randn(len(t)) * np.exp(-4 * t)
        
        if channels == 2:
            samples = samples.reshape((-1, 2))
            rev_l = fftconvolve(samples[:, 0], ir, mode='full')[:len(samples)]
            rev_r = fftconvolve(samples[:, 1], ir, mode='full')[:len(samples)]
            rev_samples = np.vstack((rev_l, rev_r)).T
        else:
            rev_samples = fftconvolve(samples, ir, mode='full')[:len(samples)]
            
        # Core fix: match RMS of Dry and Wet to maintain consistent volume
        rms_dry = np.sqrt(np.mean(samples**2))
        rms_wet = np.sqrt(np.mean(rev_samples**2))
        
        # Scale reverb volume to match original volume
        if rms_wet > 0:
            rev_samples = rev_samples * (rms_dry / rms_wet)
            
        # Adjust Dry/Wet mix using slider value
        intensity = self.reverb_intensity_slider.value() / 100.0
        wet = intensity
        dry = 1.0 - (intensity * 0.5)
        
        mixed = dry * samples + wet * rev_samples
        
        # Peak normalization to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = (mixed / max_val) * 32767.0
            
        return AudioSegment(mixed.astype(np.int16).tobytes(), frame_rate=sr, sample_width=2, channels=channels)


    # --- Newly Added: Apply final settings and generate output file ---
    def generate_final_output(self):
        # Prioritize Spatial Chorus, fallback to Non-Spatial Chorus
        base_audio_path = None
        if self.spatial_chorus_control.audio_path:
            base_audio_path = self.spatial_chorus_control.audio_path
        elif self.chorus_control.audio_path:
            base_audio_path = self.chorus_control.audio_path
            
        if not base_audio_path:
            self.final_label.setText("No chorus audio to mix.")
            return
            
        self.final_label.setText("Rendering final output...")
        QApplication.processEvents() # Prevent UI freeze
        
        try:
            audio = AudioSegment.from_wav(base_audio_path)
            
            # Check if Global Reverb is applied
            if self.global_reverb_toggle.isChecked():
                audio = self.apply_reverb_effect(audio)
                
            final_file = os.path.join(tempfile.gettempdir(), "final_master_output.wav")
            audio.export(final_file, format="wav")
            
            self.final_player.setMedia(QMediaContent(QUrl.fromLocalFile(final_file)))
            self.final_label.setText("Final Output is ready.")
            
            # Save final file path and enable save button
            self.final_audio_path = final_file
            self.save_final_button.setEnabled(True)
            
        except Exception as e:
            self.final_label.setText(f"Failed to generate final output: {str(e)}")

    def play_final_audio(self):
        if self.final_player.state() == QMediaPlayer.PlayingState:
            self.final_player.pause()
            self.final_play_button.setText("Play Final Output")
        else:
            self.final_player.play()
            self.final_play_button.setText("Pause Final Output")

    def stop_final_audio(self):
        self.final_player.stop()
        self.final_play_button.setText("Play Final Output")

    def save_audio(self):
        import shutil
        import os
        
        # 1. Check if there are converted audio files to save
        if not hasattr(self, 'original_converted_files') or not self.original_converted_files:
            self.result_label.setText("No individual converted audio to save.")
            return

        # 2. Open dialog to select directory for saving
        save_dir = QFileDialog.getExistingDirectory(
            self, 
            "Select Directory to Save All Converted Audio & Chorus",
            "" # Default path
        )
        
        # If user cancels
        if not save_dir:
            return

        saved_count = 0
        try:
            # 3. Copy and save all individually converted audio files
            for temp_file_path, target_filename, _ in self.original_converted_files:
                if temp_file_path and os.path.isfile(temp_file_path):
                    # Add 'converted_' prefix to prevent name conflicts
                    save_name = f"converted_{target_filename}" 
                    dest_path = os.path.join(save_dir, save_name)
                    shutil.copy2(temp_file_path, dest_path)
                    saved_count += 1

            # 4. Copy and save the combined Non-Spatial Chorus file
            chorus_path = self.chorus_control.audio_path
            if chorus_path and os.path.isfile(chorus_path):
                dest_path = os.path.join(save_dir, "non_spatial_chorus_mix.wav")
                shutil.copy2(chorus_path, dest_path)
                saved_count += 1
                
            self.result_label.setText(f"A total of {saved_count} files have been saved in the '{os.path.basename(save_dir)}' folder.")
            
        except Exception as e:
            self.result_label.setText(f"Failed to batch save files: {str(e)}")

    def save_spatial_audio(self):
        import shutil
        import os
        
        # 1. Check if there is spatial audio data
        if not hasattr(self, 'converted_files') or not self.converted_files:
            self.result_label.setText("No spatial audio files to save. Please run 'Apply Spatial Audio' first.")
            return

        # 2. Open dialog to select directory for saving
        save_dir = QFileDialog.getExistingDirectory(
            self, 
            "Select Directory to Save Spatial Audio & Chorus",
            "" # Default path
        )
        
        # If user cancels
        if not save_dir:
            return

        saved_count = 0
        try:
            # 3. Copy and save all individual spatial audio files
            for temp_file_path, target_filename, _ in self.converted_files:
                if temp_file_path and os.path.isfile(temp_file_path):
                    # Add 'spatial_' prefix to distinguish and prevent conflicts
                    save_name = f"spatial_{target_filename}" 
                    dest_path = os.path.join(save_dir, save_name)
                    shutil.copy2(temp_file_path, dest_path)
                    saved_count += 1

            # 4. Copy and save the combined Spatial Chorus file
            spatial_chorus_path = self.spatial_chorus_control.audio_path
            if spatial_chorus_path and os.path.isfile(spatial_chorus_path):
                dest_path = os.path.join(save_dir, "spatial_chorus_mix.wav")
                shutil.copy2(spatial_chorus_path, dest_path)
                saved_count += 1
                
            self.result_label.setText(f"A total of {saved_count} spatial audio files have been saved in the '{os.path.basename(save_dir)}' folder.")
            
        except Exception as e:
            self.result_label.setText(f"Failed to batch save spatial audio files: {str(e)}")

    def save_final_audio(self):
        import shutil
        import os
        
        # 1. Check if the rendered final file exists
        if not hasattr(self, 'final_audio_path') or not os.path.isfile(self.final_audio_path):
            self.final_label.setText("No final output to save.")
            return

        # 2. Open file save dialog (specify a single file)
        save_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Final Master Output", 
            "final_master_chorus.wav", # Default filename
            "WAV Files (*.wav)"
        )
        
        # 3. If the user specifies a path, copy and save
        if save_path:
            try:
                shutil.copy2(self.final_audio_path, save_path)
                self.final_label.setText(f"Final output saved successfully: {os.path.basename(save_path)}")
            except Exception as e:
                self.final_label.setText(f"Failed to save final output: {str(e)}")

    def update_reverb_label(self):
        val = self.reverb_intensity_slider.value()
        self.reverb_intensity_label.setText(f"Reverb Intensity: {val}%")

# ==========================================
# Outside the AudioPlayer class (no indentation)
# ==========================================
from PyQt5.QtCore import QObject, pyqtSignal
import sys
import re

class OutputInterceptor(QObject):
    output_signal = pyqtSignal(str)
    
    def write(self, text):
        # Intercepts terminal text output and emits it as a signal
        self.output_signal.emit(text)
        
    def flush(self):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioPlayer()
    window.show()
    sys.exit(app.exec_())