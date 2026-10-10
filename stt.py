import io
import wave
import numpy as np
try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


class SpeechRecognizer:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """Inisialisasi model Speech-to-Text lokal.
        
        Secara default menggunakan device='cpu' dan compute_type='int8'
        agar tidak membebani VRAM GPU yang digunakan oleh Ollama Qwen.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def load_model(self):
        if self.model is None:
            if WhisperModel is None:
                raise ImportError(
                    "faster-whisper belum terinstal. Silakan jalankan: "
                    "pip install faster-whisper"
                )
            print(f"[STT] Memuat model Faster-Whisper '{self.model_size}' di {self.device.upper()}...")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print("[STT] Model siap digunakan.\n")

    def record_until_silence(
        self,
        samplerate: int = 16000,
        threshold: float = 0.02,
        silence_duration: float = 1.5,
        max_duration: float = 15.0,
    ) -> np.ndarray:
        """Merekam audio dari mikrofon hingga jeda hening terdeteksi."""
        if sd is None:
            raise ImportError(
                "sounddevice belum terinstal. Silakan jalankan: "
                "pip install sounddevice"
            )

        print("🎤 Mendengarkan... Silakan bicara (berhenti bicara untuk memproses)...")
        chunk_duration = 0.1  # 100 ms per blok
        chunk_samples = int(samplerate * chunk_duration)
        silence_chunks_needed = int(silence_duration / chunk_duration)

        recorded_chunks = []
        silence_counter = 0
        has_spoken = False
        max_chunks = int(max_duration / chunk_duration)

        with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_samples)
                recorded_chunks.append(chunk)

                # Hitung volume audio (RMS)
                volume = np.sqrt(np.mean(chunk**2))

                if volume > threshold:
                    has_spoken = True
                    silence_counter = 0
                elif has_spoken:
                    silence_counter += 1
                    if silence_counter >= silence_chunks_needed:
                        break

        if not recorded_chunks:
            return np.array([], dtype="float32")

        audio_data = np.concatenate(recorded_chunks, axis=0).flatten()
        return audio_data

    def transcribe(self, audio_data: np.ndarray, language: str = "id") -> str:
        """Mengubah array audio menjadi teks."""
        self.load_model()
        if len(audio_data) == 0:
            return ""

        segments, _ = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=3,
            vad_filter=True,
        )

        texts = [seg.text.strip() for seg in segments]
        return " ".join(texts).strip()
