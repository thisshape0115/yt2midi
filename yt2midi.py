import subprocess
import sys

REQUIRED = ["numpy", "scipy", "mido", "tqdm", "yt-dlp", "numba"]

def install_dependencies():
    for pkg in REQUIRED:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

if __name__ == "__main__":
    install_dependencies()

import numpy as np
import mido
import argparse
import os
import tempfile
import re
from scipy.signal import firwin, filtfilt
from scipy.io.wavfile import read
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
import yt_dlp
from numba import njit


def is_url(s):
    return re.match(r'https?://', s) is not None


def download_audio(url):
    tmp = tempfile.mktemp(suffix=".wav")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': tmp.replace('.wav', ''),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return tmp


class FrequencyBand:
    def __init__(self, midi_note, sr):
        self.note = midi_note
        self.freq = 440 * (2 ** ((midi_note - 69) / 12))
        self.sr = sr
        self.filter = self.create_filter()

    def create_filter(self):
        numtaps = 2049

        nyquist = self.sr / 2

        low = self.freq * 0.95
        high = self.freq * 1.05

        if high >= nyquist - 1:
            high = nyquist - 1

        if low >= high:
            low = high * 0.5

        low = max(1, low)

        return firwin(
            numtaps,
            [low, high],
            fs=self.sr,
            pass_zero=False
        )


class AudioAnalyzer:
    def __init__(self, file_path, target_sr=44100):
        self.sr, self.audio = read(file_path)

        if self.audio.ndim > 1:
            self.audio = self.audio.mean(axis=1)

        self.audio = self.audio.astype(np.float64)

        peak = np.max(np.abs(self.audio))
        if peak > 0:
            self.audio /= peak

        if self.sr != target_sr:
            self.resample(target_sr)
            self.sr = target_sr

    # 느려짐 수정된 리샘플러
    def resample(self, target_sr):
        old_len = len(self.audio)
        new_len = int(old_len * target_sr / self.sr)

        self.audio = np.interp(
            np.linspace(0, old_len - 1, new_len),
            np.arange(old_len),
            self.audio
        )


class MidiConverter:
    def __init__(self, sr=44100):
        self.sr = sr

        self.ppqn = 4410
        self.bpm = int(round(sr / self.ppqn * 60))

        self.ticks_per_sec = sr

    def convert_events(self, all_events):
        tracks = {
            note: mido.MidiTrack()
            for note in range(128)
        }

        sorted_events = sorted(
            all_events,
            key=lambda x: x[2]
        )

        last_times = {
            note: 0
            for note in range(128)
        }

        for ev_type, note, time, velocity in tqdm(
            sorted_events,
            desc="Placing notes"
        ):
            track = tracks[note]

            ticks = int(time * self.ticks_per_sec)

            delta = max(
                0,
                ticks - last_times[note]
            )

            track.append(
                mido.Message(
                    ev_type,
                    note=note,
                    velocity=velocity,
                    time=delta
                )
            )

            last_times[note] = ticks

        return [mido.MidiTrack()] + list(tracks.values())


@njit(cache=True)
def _detect_crossings(filtered, hysteresis):
    crossings = []

    state = 0

    for i in range(len(filtered)):
        current = filtered[i]

        if state == 0 and current > hysteresis:
            state = 1
            crossings.append(i)

        elif state == 1 and current < -hysteresis:
            state = 0
            crossings.append(i)

    return crossings


def analyze_band(args):
    band, audio = args

    try:
        filtered = filtfilt(
            band.filter,
            [1.0],
            audio
        )

    except ValueError:
        return []

    crossings = _detect_crossings(
        filtered,
        0.01
    )

    events = []

    for i in range(2, len(crossings), 2):
        start = crossings[i - 2]
        end = crossings[i]

        if end - start < 2:
            continue

        segment = filtered[start:end]

        # velocity 127 방지
        velocity = min(
            126,
            max(
                1,
                int(
                    np.sqrt(
                        np.max(np.abs(segment))
                    ) * 127
                )
            )
        )

        start_time = start / band.sr
        end_time = end / band.sr

        wavelength = 1.0 / band.freq

        if end_time - start_time > 1.5 * wavelength:
            end_time = start_time + wavelength

        events.append((
            'note_on',
            band.note,
            start_time,
            velocity
        ))

        events.append((
            'note_off',
            band.note,
            end_time,
            0
        ))

    return events


def main(input_path, output_path):
    tmp_file = None

    if is_url(input_path):
        print(f"Downloading audio from: {input_path}")

        tmp_file = download_audio(input_path)
        wav_path = tmp_file

    else:
        wav_path = input_path

    audio_analyzer = AudioAnalyzer(wav_path)

    print(f"Audio SR: {audio_analyzer.sr} Hz")

    if tmp_file and os.path.exists(tmp_file):
        os.remove(tmp_file)

    converter = MidiConverter(
        sr=audio_analyzer.sr
    )

    bands = [
        FrequencyBand(
            n,
            audio_analyzer.sr
        )
        for n in tqdm(
            range(128),
            desc="Creating bands"
        )
    ]

    print("Warming up JIT compiler...")

    _detect_crossings(
        np.zeros(1, dtype=np.float64),
        0.01
    )

    num_workers = min(cpu_count(), 128)

    chunksize = max(
        1,
        128 // num_workers
    )

    args = [
        (band, audio_analyzer.audio)
        for band in bands
    ]

    print(
        f"Analyzing {len(bands)} bands "
        f"on {num_workers} threads "
        f"(chunksize={chunksize})..."
    )

    with ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:

        results = list(
            tqdm(
                executor.map(
                    analyze_band,
                    args,
                    chunksize=chunksize
                ),
                total=len(bands),
                desc="Analyzing bands"
            )
        )

    all_events = [
        ev
        for band_events in results
        for ev in band_events
    ]

    midi = mido.MidiFile(
        type=1,
        ticks_per_beat=converter.ppqn
    )

    midi.tracks = converter.convert_events(
        all_events
    )

    midi.tracks[0].append(
        mido.MetaMessage(
            'set_tempo',
            tempo=mido.bpm2tempo(
                converter.bpm
            )
        )
    )

    midi.save(output_path)

    print(
        f"\nSuccessfully saved MIDI to "
        f"{output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='audio to midi converter (optimized)'
    )

    parser.add_argument(
        'input',
        help='Input WAV file or YouTube URL'
    )

    parser.add_argument(
        'output',
        help='Output MIDI file'
    )

    args = parser.parse_args()

    main(args.input, args.output)