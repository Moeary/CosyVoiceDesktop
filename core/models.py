from typing import List, Optional, Tuple, Dict

class VoiceConfig:
    """语音配置类"""
    def __init__(self, name: str = "", mode: str = "零样本复制", 
                 prompt_text: str = "", prompt_audio: str = "", 
                 instruct_text: str = "", color: str = "#FFFF00"):
        self.name = name
        self.mode = mode
        self.prompt_text = prompt_text
        self.prompt_audio = prompt_audio
        self.instruct_text = instruct_text
        self.color = color
    
    def to_dict(self):
        return {
            'name': self.name,
            'mode': self.mode,
            'prompt_text': self.prompt_text,
            'prompt_audio': self.prompt_audio,
            'instruct_text': self.instruct_text,
            'color': self.color
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class TaskSegment:
    """任务段落类"""
    def __init__(self, index: int, text: str, voice_config: VoiceConfig,
                 mode: str = None, instruct_text: str = None, seed: int = 42):
        self.index = index
        self.text = text
        self.voice_config = voice_config
        self.mode = mode or voice_config.mode
        self.instruct_text = instruct_text or voice_config.instruct_text
        self.seed = seed  # 随机种子
        self.run_count = 0
        # 二维结构：versions[版本号][片段号] = 文件路径
        self.versions: List[List[str]] = []  # [[v1_1, v1_2], [v2_1, v2_2, v2_3]]
        self.current_version = 0  # 当前选中的版本
        self.current_segment = 0  # 当前选中的片段
        self.current_audio: Optional[str] = None
    
    def add_version(self, files: List[str]):
        """添加新版本的音频文件列表"""
        if files:
            self.versions.append(files)
            self.run_count = len(self.versions)
            # 默认选择最新版本的第一个片段
            self.current_version = len(self.versions) - 1
            self.current_segment = 0
            self.current_audio = files[0]
    
    def get_all_audio_options(self) -> List[Tuple[int, int, str]]:
        """获取所有音频选项 (版本号, 片段号, 文件路径)"""
        options = []
        for ver_idx, version_files in enumerate(self.versions):
            for seg_idx, filepath in enumerate(version_files):
                options.append((ver_idx + 1, seg_idx + 1, filepath))
        return options
    
    def set_audio(self, version: int, segment: int):
        """设置当前播放的音频"""
        ver_idx = version - 1
        seg_idx = segment - 1
        if 0 <= ver_idx < len(self.versions) and 0 <= seg_idx < len(self.versions[ver_idx]):
            self.current_version = ver_idx
            self.current_segment = seg_idx
            self.current_audio = self.versions[ver_idx][seg_idx]
            return True
        return False
    
    def get_latest_audio(self) -> Optional[str]:
        """获取最新生成的音频文件"""
        if self.versions and self.versions[-1]:
            return self.versions[-1][0]
        return None
