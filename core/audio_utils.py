import io
from typing import Iterable, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from scipy.io import wavfile as scipy_wavfile
except Exception:
    scipy_wavfile = None


def speech_to_numpy(speech) -> np.ndarray:
    """Convert torch/numpy speech tensor to float32 ndarray with shape [channels, samples]."""
    if speech is None:
        raise ValueError("speech is None")

    if hasattr(speech, "detach"):
        speech = speech.detach()
    if hasattr(speech, "cpu"):
        speech = speech.cpu()
    if hasattr(speech, "numpy"):
        speech = speech.numpy()

    audio = np.asarray(speech, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    if audio.ndim != 2:
        raise ValueError(f"Unexpected speech ndim={audio.ndim}, expected 1 or 2.")

    # Normalize to [channels, samples]
    if audio.shape[0] > audio.shape[1] and audio.shape[1] <= 8:
        audio = audio.T

    return np.ascontiguousarray(audio, dtype=np.float32)


def concat_speeches(speeches: Iterable) -> np.ndarray:
    chunks = [speech_to_numpy(item) for item in speeches]
    if not chunks:
        raise ValueError("No speeches to concatenate")

    first_channels = chunks[0].shape[0]
    for idx, chunk in enumerate(chunks):
        if chunk.shape[0] != first_channels:
            raise ValueError(
                f"Channel mismatch at chunk {idx}: {chunk.shape[0]} vs {first_channels}"
            )

    return np.concatenate(chunks, axis=1)


def load_wav(path: str) -> Tuple[np.ndarray, int]:
    if sf is not None:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return np.ascontiguousarray(audio.T, dtype=np.float32), int(sample_rate)

    if scipy_wavfile is not None:
        sample_rate, audio = scipy_wavfile.read(path)
        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=1)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        return np.ascontiguousarray(audio.T, dtype=np.float32), int(sample_rate)

    raise RuntimeError("缺少音频读写依赖，请安装 soundfile 或 scipy")


def _to_int16(data: np.ndarray) -> np.ndarray:
    clipped = np.clip(data, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def save_wav_file(path: str, audio: np.ndarray, sample_rate: int):
    data = speech_to_numpy(audio).T
    if sf is not None:
        sf.write(path, data, sample_rate, subtype="PCM_16")
        return

    if scipy_wavfile is not None:
        scipy_wavfile.write(path, sample_rate, _to_int16(data))
        return

    raise RuntimeError("缺少音频读写依赖，请安装 soundfile 或 scipy")


def save_wav_buffer(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    buffer = io.BytesIO()
    data = speech_to_numpy(audio).T
    if sf is not None:
        sf.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer

    if scipy_wavfile is not None:
        scipy_wavfile.write(buffer, sample_rate, _to_int16(data))
        buffer.seek(0)
        return buffer

    raise RuntimeError("缺少音频读写依赖，请安装 soundfile 或 scipy")
