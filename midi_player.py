"""
🎹 HyperMIDI — Ultra-high-performance MIDI Player & Piano Roll Scroller
Handles tens of millions of notes via virtualized rendering + spatial indexing.

Requirements: PyQt5, mido, pygame, numpy
  pip install PyQt5 mido pygame numpy
"""

import sys, os, time, math, threading
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import mido
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QSizePolicy,
    QScrollBar, QStatusBar, QToolBar, QComboBox, QFrame, QProgressBar, QShortcut
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPainterPath, QPalette, QKeySequence
)

# ─────────────────────────────────────────────────────────────
#  DATA MODEL
# ─────────────────────────────────────────────────────────────

@dataclass
class MidiNote:
    pitch:    int
    start:    float
    end:      float
    velocity: int
    channel:  int
    track:    int

    @property
    def duration(self) -> float:
        return max(0.01, self.end - self.start)


CHANNEL_COLORS = [
    QColor("#FF6B6B"), QColor("#4ECDC4"), QColor("#45B7D1"),
    QColor("#96CEB4"), QColor("#FFEAA7"), QColor("#DDA0DD"),
    QColor("#98D8C8"), QColor("#F7DC6F"), QColor("#BB8FCE"),
    QColor("#85C1E9"), QColor("#82E0AA"), QColor("#F0B27A"),
    QColor("#AEB6BF"), QColor("#F1948A"), QColor("#73C6B6"),
    QColor("#7FB3D3"),
]

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ─────────────────────────────────────────────────────────────
#  MIDI LOADER
# ─────────────────────────────────────────────────────────────

class MidiLoader(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, float, dict)
    error    = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.progress.emit(5, "파일 읽는 중…")
            mid = mido.MidiFile(self.path)
            ticks_per_beat = mid.ticks_per_beat

            self.progress.emit(15, "템포 맵 구축 중…")
            tempo_map = []
            for track in mid.tracks:
                abs_tick = 0
                for msg in track:
                    abs_tick += msg.time
                    if msg.type == 'set_tempo':
                        tempo_map.append((abs_tick, msg.tempo))
            tempo_map.sort(key=lambda x: x[0])
            if not tempo_map:
                tempo_map = [(0, 500000)]

            def ticks_to_seconds(tick):
                seconds = 0.0
                prev_tick, prev_tempo = 0, 500000
                for tm_tick, tm_tempo in tempo_map:
                    if tm_tick >= tick:
                        break
                    delta = min(tm_tick, tick) - prev_tick
                    seconds += delta / ticks_per_beat * (prev_tempo / 1_000_000)
                    prev_tick, prev_tempo = tm_tick, tm_tempo
                delta = tick - prev_tick
                seconds += delta / ticks_per_beat * (prev_tempo / 1_000_000)
                return seconds

            self.progress.emit(30, "노트 파싱 중…")
            notes = []
            active = {}
            total_tracks = len(mid.tracks)

            for t_idx, track in enumerate(mid.tracks):
                pct = 30 + int(60 * t_idx / max(total_tracks, 1))
                self.progress.emit(pct, f"트랙 {t_idx+1}/{total_tracks} 파싱 중…")
                abs_tick = 0
                for msg in track:
                    abs_tick += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        active[(msg.channel, msg.note, t_idx)] = (abs_tick, msg.velocity)
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        key = (msg.channel, msg.note, t_idx)
                        if key in active:
                            start_tick, vel = active.pop(key)
                            notes.append(MidiNote(
                                pitch=msg.note,
                                start=ticks_to_seconds(start_tick),
                                end=ticks_to_seconds(abs_tick),
                                velocity=vel,
                                channel=msg.channel,
                                track=t_idx,
                            ))

            for (ch, pitch, t_idx), (start_tick, vel) in active.items():
                notes.append(MidiNote(
                    pitch=pitch,
                    start=ticks_to_seconds(start_tick),
                    end=ticks_to_seconds(start_tick) + 0.1,
                    velocity=vel, channel=ch, track=t_idx,
                ))

            self.progress.emit(92, "노트 정렬 중…")
            notes.sort(key=lambda n: n.start)
            duration = max((n.end for n in notes), default=0.0)
            meta = {
                'filename': os.path.basename(self.path),
                'note_count': len(notes),
                'track_count': total_tracks,
                'duration': duration,
            }
            self.progress.emit(100, "완료!")
            self.finished.emit(notes, duration, meta)
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────────
#  SPATIAL INDEX  — O(1) visible note lookup
# ─────────────────────────────────────────────────────────────

class SpatialIndex:
    def __init__(self, notes: List[MidiNote], bucket_secs: float = 0.5):
        self.bucket_secs = bucket_secs
        self.buckets: dict = defaultdict(list)
        for note in notes:
            b0 = int(note.start / bucket_secs)
            b1 = int(note.end   / bucket_secs)
            for b in range(b0, b1 + 1):
                self.buckets[b].append(note)

    def query(self, t0: float, t1: float) -> List[MidiNote]:
        b0 = int(t0 / self.bucket_secs)
        b1 = int(t1 / self.bucket_secs)
        seen = set()
        result = []
        for b in range(b0, b1 + 1):
            for note in self.buckets.get(b, []):
                nid = id(note)
                if nid not in seen and note.end >= t0 and note.start <= t1:
                    seen.add(nid)
                    result.append(note)
        return result


# ─────────────────────────────────────────────────────────────
#  PIANO ROLL WIDGET
# ─────────────────────────────────────────────────────────────

class PianoRollWidget(QWidget):
    timeChanged = pyqtSignal(float)

    PIANO_W   = 52
    RULER_H   = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.notes:    List[MidiNote] = []
        self.index:    Optional[SpatialIndex] = None
        self.duration: float = 0.0

        self.view_start:  float = 0.0
        self.px_per_sec:  float = 150.0
        self.note_height: float = 8.0

        self.playhead: float = 0.0

        self._drag_x:   Optional[int] = None
        self._drag_vs:  float = 0.0
        self._hovered:  Optional[MidiNote] = None

        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    # ── public API ───────────────────────────────────────────

    def load(self, notes, duration):
        self.notes = notes
        self.duration = duration
        self.index = SpatialIndex(notes)
        self.view_start = 0.0
        self.update()

    def set_playhead(self, t: float):
        self.playhead = t
        vis = self._visible_secs()
        if t > self.view_start + vis * 0.85 or t < self.view_start:
            self.view_start = max(0.0, t - vis * 0.15)
        self.update()

    def set_view_start(self, t: float):
        self.view_start = max(0.0, min(t, max(0.0, self.duration - self._visible_secs())))
        self.update()

    def zoom_x(self, factor, pivot_px=None):
        old = self.px_per_sec
        self.px_per_sec = max(5.0, min(3000.0, self.px_per_sec * factor))
        if pivot_px is not None:
            pt = self.view_start + (pivot_px - self.PIANO_W) / old
            self.view_start = max(0.0, pt - (pivot_px - self.PIANO_W) / self.px_per_sec)
        self.update()

    def zoom_y(self, factor):
        self.note_height = max(2.0, min(32.0, self.note_height * factor))
        self.update()

    # ── helpers ──────────────────────────────────────────────

    def _visible_secs(self):
        return max(0.001, (self.width() - self.PIANO_W) / self.px_per_sec)

    def _t_to_x(self, t):
        return self.PIANO_W + (t - self.view_start) * self.px_per_sec

    def _pitch_to_y(self, p):
        return self.height() - (p + 1) * self.note_height

    def _x_to_t(self, x):
        return self.view_start + (x - self.PIANO_W) / self.px_per_sec

    # ── paint ────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor("#0D0F14"))

        # Black-key row shading
        black_keys = {1, 3, 6, 8, 10}
        bk_color   = QColor("#0A0B10")
        nh = self.note_height
        for pitch in range(128):
            if pitch % 12 in black_keys:
                y = int(self._pitch_to_y(pitch))
                p.fillRect(self.PIANO_W, y, w - self.PIANO_W, max(1, int(nh)), bk_color)

        # Octave lines
        p.setPen(QPen(QColor("#1E2230"), 1))
        for pitch in range(0, 128, 12):
            y = int(self._pitch_to_y(pitch + 12))
            p.drawLine(self.PIANO_W, y, w, y)

        # Beat grid
        if self.px_per_sec > 30:
            beat = 1.0
            if self.px_per_sec > 300: beat = 0.25
            elif self.px_per_sec > 100: beat = 0.5
            p.setPen(QPen(QColor("#181B26"), 1))
            t = math.floor(self.view_start / beat) * beat
            end_t = self.view_start + self._visible_secs() + beat
            while t < end_t:
                x = int(self._t_to_x(t))
                if self.PIANO_W <= x < w:
                    p.drawLine(x, self.RULER_H, x, h)
                t += beat

        # Notes
        visible = self.index.query(self.view_start - 1, self.view_start + self._visible_secs() + 1) \
            if self.index else []

        p.setRenderHint(QPainter.Antialiasing, nh >= 8)
        inh = max(1, int(nh) - 1)

        for note in visible:
            x1 = int(self._t_to_x(note.start))
            x2 = int(self._t_to_x(note.end))
            y  = int(self._pitch_to_y(note.pitch))
            nw = max(2, x2 - x1)

            bc = CHANNEL_COLORS[note.channel % 16]
            v  = note.velocity / 127.0
            r  = int(bc.red()   * (0.35 + 0.65 * v))
            g  = int(bc.green() * (0.35 + 0.65 * v))
            b  = int(bc.blue()  * (0.35 + 0.65 * v))

            if note is self._hovered:
                p.fillRect(x1-1, y-1, nw+2, inh+2, QColor(255,255,255,70))

            p.fillRect(x1, y, nw, inh, QColor(r, g, b))

            # Top highlight
            if inh >= 4:
                p.fillRect(x1, y, nw, 1, QColor(min(255,r+55), min(255,g+55), min(255,b+55)))

        # Piano keyboard
        self._draw_piano(p, h)

        # Playhead
        px = int(self._t_to_x(self.playhead))
        if self.PIANO_W <= px < w:
            p.setPen(QPen(QColor("#FF4757"), 2))
            p.drawLine(px, self.RULER_H, px, h)
            p.setBrush(QBrush(QColor("#FF4757")))
            p.setPen(Qt.NoPen)
            path = QPainterPath()
            path.moveTo(px-6, self.RULER_H)
            path.lineTo(px+6, self.RULER_H)
            path.lineTo(px,   self.RULER_H + 10)
            path.closeSubpath()
            p.drawPath(path)

        # Time ruler
        self._draw_ruler(p, w)

        p.end()

    def _draw_piano(self, p, h):
        pw = self.PIANO_W
        black_keys = {1, 3, 6, 8, 10}
        nh = self.note_height
        inh = max(1, int(nh))
        font = QFont("Courier New", max(6, int(nh * 0.65)))
        p.setFont(font)

        for pitch in range(128):
            y = int(self._pitch_to_y(pitch))
            is_black = pitch % 12 in black_keys
            color = QColor("#1A1A2E") if is_black else QColor("#DCDCE8")
            bw = pw if is_black else pw - 2
            p.fillRect(0, y, bw, inh - 1, color)

            if pitch % 12 == 0 and nh >= 7:
                p.setPen(QColor("#777799"))
                p.drawText(3, y + inh - 2, f"C{pitch//12 - 1}")

        p.fillRect(pw-2, 0, 2, h, QColor("#2A2D3E"))

    def _draw_ruler(self, p, w):
        rh = self.RULER_H
        p.fillRect(self.PIANO_W, 0, w - self.PIANO_W, rh, QColor("#10121A"))
        p.setPen(QPen(QColor("#2A2D3E"), 1))
        p.drawLine(self.PIANO_W, rh, w, rh)

        step = 1.0
        if self.px_per_sec > 300: step = 0.25
        elif self.px_per_sec > 100: step = 0.5
        elif self.px_per_sec < 30: step = 5.0

        font = QFont("Courier New", 8)
        p.setFont(font)
        t = math.floor(self.view_start / step) * step
        end_t = self.view_start + self._visible_secs() + step
        while t < end_t:
            x = int(self._t_to_x(t))
            if self.PIANO_W <= x < w:
                p.setPen(QPen(QColor("#3A3D50"), 1))
                p.drawLine(x, rh-7, x, rh)
                mins = int(t) // 60
                secs = t % 60
                p.setPen(QColor("#7880AA"))
                p.drawText(x+2, 13, f"{mins}:{secs:05.2f}")
            t += step

    # ── mouse ────────────────────────────────────────────────

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and e.x() > self.PIANO_W:
            t = max(0.0, min(self._x_to_t(e.x()), self.duration))
            self.playhead = t
            self.timeChanged.emit(t)
            self.update()
        elif e.button() in (Qt.MiddleButton, Qt.RightButton):
            self._drag_x  = e.x()
            self._drag_vs = self.view_start
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag_x is not None:
            dt = -(e.x() - self._drag_x) / self.px_per_sec
            self.view_start = max(0.0, self._drag_vs + dt)
            self.update()
        elif self.index and e.x() > self.PIANO_W:
            t = self._x_to_t(e.x())
            pitch = int((self.height() - e.y()) / self.note_height)
            cands = self.index.query(t - 0.02, t + 0.02)
            found = next((n for n in cands if n.pitch == pitch and n.start <= t <= n.end), None)
            if found != self._hovered:
                self._hovered = found
                self.update()
                if found:
                    nm = NOTE_NAMES[found.pitch % 12]
                    oc = found.pitch // 12 - 1
                    self.setToolTip(f"{nm}{oc}  vel={found.velocity}  ch={found.channel+1}  dur={found.duration:.3f}s")
                else:
                    self.setToolTip("")

    def mouseReleaseEvent(self, e):
        self._drag_x = None
        self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, e):
        delta = e.angleDelta().y()
        mods  = e.modifiers()
        if mods & Qt.ControlModifier:
            self.zoom_x(1.15 if delta > 0 else 1/1.15, e.x())
        elif mods & Qt.ShiftModifier:
            self.zoom_y(1.15 if delta > 0 else 1/1.15)
        else:
            dt = -delta / self.px_per_sec * 0.3
            self.view_start = max(0.0, self.view_start + dt)
            self.update()


# ─────────────────────────────────────────────────────────────
#  PLAYBACK ENGINE
# ─────────────────────────────────────────────────────────────

class PlaybackEngine(QThread):
    tick    = pyqtSignal(float)
    stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._notes = []
        self._playing = False
        self._seek_time = 0.0
        self._start_wall = 0.0
        self._tempo_scale = 1.0

    def load(self, notes):
        self._notes = notes

    def play(self, from_time=0.0):
        self._seek_time = from_time
        self._playing   = True
        if not self.isRunning():
            self.start()
        else:
            self._start_wall = time.perf_counter() - from_time / self._tempo_scale

    def pause(self):
        self._playing = False

    def seek(self, t):
        self._seek_time  = t
        self._start_wall = time.perf_counter() - t / self._tempo_scale

    def stop(self):
        self._playing = False
        self.quit()

    def set_tempo_scale(self, s):
        self._tempo_scale = s

    def run(self):
        self._start_wall = time.perf_counter() - self._seek_time / self._tempo_scale
        while self._playing:
            t = (time.perf_counter() - self._start_wall) * self._tempo_scale
            self.tick.emit(t)
            time.sleep(0.016)
        self.stopped.emit()


# ─────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────

DARK_SS = """
QMainWindow,QWidget{background:#0D0F14;color:#C0C4D8;}
QToolBar{background:#12141C;border-bottom:1px solid #1E2230;spacing:3px;padding:3px;}
QPushButton{
  background:#1E2230;color:#C0C4D8;border:1px solid #2A2D3E;
  border-radius:4px;padding:4px 12px;font-family:'Courier New';font-size:12px;
}
QPushButton:hover{background:#262A3C;border-color:#4A5070;}
QPushButton:pressed{background:#161920;}
QPushButton#play_btn{background:#1A3328;border-color:#2ECC71;color:#2ECC71;font-weight:bold;}
QPushButton#play_btn:hover{background:#1E3C2E;}
QPushButton#stop_btn{background:#321A1C;border-color:#E74C3C;color:#E74C3C;}
QSlider::groove:horizontal{background:#1A1D28;height:4px;border-radius:2px;}
QSlider::handle:horizontal{
  background:#4A90D9;width:12px;height:12px;margin:-4px 0;border-radius:6px;
}
QSlider::sub-page:horizontal{background:#4A90D9;border-radius:2px;}
QLabel{color:#8890AA;font-family:'Courier New';font-size:11px;}
QLabel#meta_lbl{color:#DDE0F0;font-size:12px;font-weight:bold;}
QStatusBar{background:#090A0F;color:#555770;font-family:'Courier New';font-size:10px;}
QProgressBar{background:#12141C;border:1px solid #1E2230;border-radius:3px;color:#C0C4D8;}
QProgressBar::chunk{background:#4A90D9;}
QComboBox{
  background:#1E2230;color:#C0C4D8;border:1px solid #2A2D3E;
  border-radius:3px;padding:2px 6px;font-family:'Courier New';font-size:11px;
}
QScrollBar:horizontal{background:#0D0F14;height:10px;}
QScrollBar::handle:horizontal{background:#2A2D3E;border-radius:4px;min-width:20px;}
QScrollBar::handle:horizontal:hover{background:#3A3D50;}
QScrollBar::add-line,QScrollBar::sub-line{width:0;}
"""


class HyperMIDI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎹  HyperMIDI  —  Ultra Piano Roll")
        self.resize(1280, 720)
        self.setStyleSheet(DARK_SS)

        self.notes:    List[MidiNote] = []
        self.duration: float = 0.0
        self.meta:     dict  = {}
        self._seeking: bool  = False

        self.engine = PlaybackEngine()
        self.engine.tick.connect(self._on_tick)
        self.engine.stopped.connect(self._on_stopped)

        self._build_ui()
        self.status.showMessage("MIDI 파일을 열어 시작하세요  —  File › Open  /  Ctrl+O")

    def _build_ui(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))

        def btn(text, tip, slot, name=None):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFixedHeight(26)
            if name: b.setObjectName(name)
            b.clicked.connect(slot)
            tb.addWidget(b)
            return b

        btn("📂  열기", "MIDI 파일 열기 (Ctrl+O)", self.open_file)
        tb.addSeparator()
        self.play_btn = btn("▶  재생", "재생/일시정지 (Space)", self.toggle_play, "play_btn")
        self.stop_btn = btn("■  정지", "정지 (Esc)", self.stop_playback, "stop_btn")
        tb.addSeparator()

        tb.addWidget(QLabel("  속도: "))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25×","0.5×","0.75×","1.0×","1.25×","1.5×","2.0×","4.0×"])
        self.speed_combo.setCurrentIndex(3)
        self.speed_combo.setFixedWidth(68)
        self.speed_combo.currentIndexChanged.connect(self._on_speed)
        tb.addWidget(self.speed_combo)
        tb.addSeparator()

        tb.addWidget(QLabel("  🔍 시간축: "))
        self.zx = QSlider(Qt.Horizontal)
        self.zx.setRange(1, 220); self.zx.setValue(65); self.zx.setFixedWidth(110)
        self.zx.valueChanged.connect(lambda v: self._set_zoom_x(v))
        tb.addWidget(self.zx)

        tb.addWidget(QLabel("  피치축: "))
        self.zy = QSlider(Qt.Horizontal)
        self.zy.setRange(2, 80); self.zy.setValue(8); self.zy.setFixedWidth(80)
        self.zy.valueChanged.connect(lambda v: self._set_zoom_y(v))
        tb.addWidget(self.zy)
        tb.addSeparator()

        self.meta_lbl = QLabel("파일 없음")
        self.meta_lbl.setObjectName("meta_lbl")
        tb.addWidget(QLabel("  "))
        tb.addWidget(self.meta_lbl)

        # Central
        cw = QWidget(); self.setCentralWidget(cw)
        vl = QVBoxLayout(cw); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)

        self.prog_bar = QProgressBar(); self.prog_bar.setFixedHeight(16); self.prog_bar.hide()
        self.prog_lbl = QLabel(); self.prog_lbl.setAlignment(Qt.AlignCenter); self.prog_lbl.hide()
        vl.addWidget(self.prog_bar)
        vl.addWidget(self.prog_lbl)

        self.roll = PianoRollWidget()
        self.roll.timeChanged.connect(self._on_seek_click)
        vl.addWidget(self.roll, 1)

        # Bottom bar
        bf = QFrame(); bf.setFixedHeight(30)
        bf.setStyleSheet("background:#090A0F;border-top:1px solid #181A24;")
        bh = QHBoxLayout(bf); bh.setContentsMargins(6,3,6,3)

        self.time_lbl = QLabel("0:00.00 / 0:00.00")
        self.time_lbl.setFixedWidth(140)
        bh.addWidget(self.time_lbl)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 10000)
        self.seek_slider.sliderPressed.connect(lambda: setattr(self, '_seeking', True))
        self.seek_slider.sliderReleased.connect(self._seek_release)
        bh.addWidget(self.seek_slider)

        self.note_count_lbl = QLabel("노트: —")
        self.note_count_lbl.setFixedWidth(120)
        self.note_count_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bh.addWidget(self.note_count_lbl)
        vl.addWidget(bf)

        self.h_scroll = QScrollBar(Qt.Horizontal)
        self.h_scroll.setRange(0, 10000)
        self.h_scroll.valueChanged.connect(lambda v: self.roll.set_view_start(v / 100))
        vl.addWidget(self.h_scroll)

        self.status = QStatusBar(); self.setStatusBar(self.status)

        # Shortcuts
        QShortcut(QKeySequence("Space"),  self, self.toggle_play)
        QShortcut(QKeySequence("Escape"), self, self.stop_playback)
        QShortcut(QKeySequence("Ctrl+O"), self, self.open_file)
        QShortcut(QKeySequence(Qt.Key_Left),  self, lambda: self._nudge(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self._nudge(+1))
        QShortcut(QKeySequence("Ctrl+="  ), self, lambda: self.roll.zoom_x(1.2))
        QShortcut(QKeySequence("Ctrl+-"  ), self, lambda: self.roll.zoom_x(1/1.2))

    # ── file ─────────────────────────────────────────────────

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "MIDI 파일 열기", "", "MIDI Files (*.mid *.midi);;All (*)")
        if path: self._load_file(path)

    def _load_file(self, path):
        self.stop_playback()
        self.prog_bar.setValue(0); self.prog_bar.show()
        self.prog_lbl.show()
        self.loader = MidiLoader(path)
        self.loader.progress.connect(lambda p, m: (self.prog_bar.setValue(p), self.prog_lbl.setText(m)))
        self.loader.finished.connect(self._on_loaded)
        self.loader.error.connect(lambda e: (self.prog_bar.hide(), self.prog_lbl.hide(),
                                              self.status.showMessage(f"❌ 오류: {e}")))
        self.loader.start()

    def _on_loaded(self, notes, duration, meta):
        self.notes = notes; self.duration = duration; self.meta = meta
        self.roll.load(notes, duration)
        self.engine.load(notes)
        self.meta_lbl.setText(meta['filename'])
        self.note_count_lbl.setText(f"노트: {meta['note_count']:,}")
        self.h_scroll.setRange(0, max(1, int(duration * 100)))
        self.prog_bar.hide(); self.prog_lbl.hide()
        self.status.showMessage(
            f"✅  {meta['filename']}  |  노트 {meta['note_count']:,}개  |  "
            f"트랙 {meta['track_count']}개  |  길이 {self._fmt(duration)}"
        )

    # ── playback ─────────────────────────────────────────────

    def toggle_play(self):
        if self.engine.isRunning() and self.engine._playing:
            self.engine.pause()
            self.play_btn.setText("▶  재생")
        else:
            if not self.notes: return
            self.engine.play(from_time=self.roll.playhead)
            self.play_btn.setText("⏸  일시정지")

    def stop_playback(self):
        self.engine.pause()
        self.play_btn.setText("▶  재생")

    def _on_tick(self, t):
        if t > self.duration:
            self.engine.pause(); self.play_btn.setText("▶  재생"); return
        self.roll.set_playhead(t)
        self.time_lbl.setText(f"{self._fmt(t)} / {self._fmt(self.duration)}")
        if not self._seeking and self.duration > 0:
            self.seek_slider.setValue(int(t / self.duration * 10000))
        if self.duration > 0:
            self.h_scroll.blockSignals(True)
            self.h_scroll.setValue(int(self.roll.view_start * 100))
            self.h_scroll.blockSignals(False)

    def _on_stopped(self):
        self.play_btn.setText("▶  재생")

    def _on_seek_click(self, t):
        self.engine.seek(t)
        self.time_lbl.setText(f"{self._fmt(t)} / {self._fmt(self.duration)}")

    def _seek_release(self):
        self._seeking = False
        if self.duration > 0:
            t = self.seek_slider.value() / 10000 * self.duration
            self.roll.set_playhead(t); self.engine.seek(t)

    def _nudge(self, d):
        t = max(0.0, self.roll.playhead + d)
        self.roll.set_playhead(t); self.engine.seek(t)

    # ── zoom ─────────────────────────────────────────────────

    def _set_zoom_x(self, v):
        self.roll.px_per_sec = 8 * (1.04 ** v)
        self.roll.update()

    def _set_zoom_y(self, v):
        self.roll.note_height = v * 0.45 + 1.5
        self.roll.update()

    def _on_speed(self, idx):
        speeds = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]
        self.engine.set_tempo_scale(speeds[idx])

    # ── utils ─────────────────────────────────────────────────

    def _fmt(self, t):
        m = int(t) // 60; s = t % 60
        return f"{m}:{s:05.2f}"

    def closeEvent(self, _):
        self.engine.stop(); self.engine.wait(500)


# ─────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("HyperMIDI")
    app.setStyle("Fusion")

    pal = app.palette()
    dark = {
        pal.Window:          "#0D0F14", pal.WindowText:    "#C0C4D8",
        pal.Base:            "#12141C", pal.AlternateBase: "#1A1D28",
        pal.Text:            "#C0C4D8", pal.Button:        "#1E2230",
        pal.ButtonText:      "#C0C4D8", pal.Highlight:     "#4A90D9",
        pal.HighlightedText: "#FFFFFF", pal.Link:          "#4A90D9",
        pal.ToolTipBase:     "#1E2230", pal.ToolTipText:   "#C0C4D8",
    }
    for role, hex_ in dark.items():
        pal.setColor(role, QColor(hex_))
    app.setPalette(pal)

    win = HyperMIDI()
    win.show()

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        win._load_file(sys.argv[1])

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
