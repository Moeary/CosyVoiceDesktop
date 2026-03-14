import importlib.util
import os
from types import ModuleType
from typing import Generator
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .audio_utils import speech_to_numpy


def _load_module_from_path(module_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cosyvoice_onnx_inference_pure", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inject_nvidia_bins_to_path():
    """
    Ensure CUDA/cuDNN DLL directories from pip nvidia wheels are visible to ORT.
    This mainly targets Windows + pixi virtual env layouts.
    """
    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    existed = set(p.lower() for p in path_parts if p)

    candidates = []

    env_prefix = os.environ.get("CONDA_PREFIX")
    if env_prefix:
        site_packages = Path(env_prefix) / "Lib" / "site-packages"
        if site_packages.exists():
            nvidia_root = site_packages / "nvidia"
            if nvidia_root.exists():
                for bin_dir in nvidia_root.glob("*/bin"):
                    if bin_dir.is_dir():
                        candidates.append(str(bin_dir.resolve()))

        # Fallback: many users already have a working CUDA/cuDNN DLL set in default torch env.
        # If onnx-gpu env misses runtime bins, reuse that path to avoid manual PATH tweaks.
        default_torch_lib = (
            Path(env_prefix).parent / "default" / "Lib" / "site-packages" / "torch" / "lib"
        )
        if default_torch_lib.exists():
            candidates.append(str(default_torch_lib.resolve()))

    for entry in candidates:
        if entry.lower() not in existed:
            path_parts.insert(0, entry)
            existed.add(entry.lower())

    os.environ["PATH"] = os.pathsep.join(path_parts)


class OnnxRuntimeCosyVoice:
    """Adapter that exposes CosyVoice-like methods for ONNX Runtime backend."""

    backend = "onnx"

    def __init__(self, model_dir: str, use_fp16: bool = True):
        _inject_nvidia_bins_to_path()

        self.model_dir = os.path.abspath(model_dir)
        self.sample_rate = 24000

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "未检测到 CUDAExecutionProvider，当前 ONNX Runtime 仅可用: "
                f"{providers}\n"
                "请使用 onnx-gpu 环境运行（pixi run -e onnx-gpu start），并确保 CUDA/cuDNN DLL 可被加载。"
            )

        script_path = os.path.join(self.model_dir, "onnx", "scripts", "onnx_inference_pure.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(
                f"ONNX 推理脚本不存在: {script_path}\n"
                "请将 ayousanz/cosy-voice3-onnx 下载到 CosyVoice 模型目录下的 onnx 子目录。"
            )

        module = _load_module_from_path(script_path)
        engine_cls = getattr(module, "PureOnnxCosyVoice3", None)
        if engine_cls is None:
            raise RuntimeError(f"在 {script_path} 中未找到 PureOnnxCosyVoice3 类")

        self._engine = engine_cls(model_dir=self.model_dir, use_fp16=use_fp16)
        self.sample_rate = int(getattr(self._engine, "sample_rate", self.sample_rate))

    def close(self):
        self._engine = None

    def _inference_once(self, text: str, prompt_text: str, prompt_wav: str) -> np.ndarray:
        if not prompt_text:
            raise ValueError("ONNX 后端要求 prompt_text 不能为空")
        if not prompt_wav or not os.path.exists(prompt_wav):
            raise FileNotFoundError(f"参考音频不存在: {prompt_wav}")

        result = self._engine.inference(
            text=text,
            prompt_wav=prompt_wav,
            prompt_text=prompt_text,
        )

        # 兼容不同脚本版本的返回格式：
        # - 旧版: (sample_rate, audio)
        # - 当前你使用的版本: audio
        if isinstance(result, tuple):
            if len(result) >= 2 and np.isscalar(result[0]):
                self.sample_rate = int(result[0])
                speech = result[1]
            elif len(result) >= 1:
                speech = result[0]
            else:
                raise RuntimeError("ONNX inference 返回了空 tuple")
        else:
            speech = result

        return speech_to_numpy(speech)

    def inference_zero_shot(
        self,
        text: str,
        prompt_text: str,
        prompt_wav: str,
        zero_shot_spk_id: str = "",
        stream: bool = False,
        speed: float = 1.0,
        text_frontend: bool = True,
    ) -> Generator[dict, None, None]:
        del zero_shot_spk_id, stream, speed, text_frontend
        speech = self._inference_once(text=text, prompt_text=prompt_text, prompt_wav=prompt_wav)
        yield {"tts_speech": speech}

    def inference_instruct2(
        self,
        text: str,
        instruct_text: str,
        prompt_wav: str,
        zero_shot_spk_id: str = "",
        stream: bool = False,
        speed: float = 1.0,
        text_frontend: bool = True,
    ) -> Generator[dict, None, None]:
        del zero_shot_spk_id, stream, speed, text_frontend
        speech = self._inference_once(text=text, prompt_text=instruct_text, prompt_wav=prompt_wav)
        yield {"tts_speech": speech}

    def inference_cross_lingual(
        self,
        text: str,
        prompt_wav: str,
        zero_shot_spk_id: str = "",
        stream: bool = False,
        speed: float = 1.0,
        text_frontend: bool = True,
        prompt_text: str = "",
    ) -> Generator[dict, None, None]:
        del zero_shot_spk_id, stream, speed, text_frontend
        speech = self._inference_once(text=text, prompt_text=prompt_text, prompt_wav=prompt_wav)
        yield {"tts_speech": speech}
