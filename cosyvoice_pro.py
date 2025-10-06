#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice Desktop Application
支持任务计划、单段落推理、音频管理、模型卸载等高级功能
"""

import sys
import os
import json
import datetime
import subprocess
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLabel, QMessageBox, QFileDialog, QHeaderView,
    QMenu, QAction, QColorDialog, QSplitter, QProgressBar, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt5.QtGui import QFont, QTextCursor, QTextCharFormat, QColor, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

from qfluentwidgets import (
    PushButton, PrimaryPushButton, TextEdit, TableWidget, LineEdit,
    ComboBox, FluentIcon, InfoBar, InfoBarPosition, setTheme, Theme,
    NavigationInterface, NavigationItemPosition, FluentWindow, SubtitleLabel,
    BodyLabel, StrongBodyLabel, SimpleCardWidget, PlainTextEdit, ToolButton,
    TransparentToolButton
)


# ==================== 数据模型 ====================

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


# ==================== 自定义控件 ====================

class CustomTextEdit(TextEdit):
    """自定义文本编辑器"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.voice_configs: Dict[str, VoiceConfig] = {}
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 快捷指令定义（基于CosyVoice 2文档）
        self.quick_tags = {
            'strong': {'tag': '<strong>{}</strong>', 'name': '强调', 'shortcut': 'Alt+S'},
            'laughter': {'tag': '<laughter>{}</laughter>', 'name': '笑声', 'shortcut': 'Alt+L'},
            'breath': {'tag': '[breath]', 'name': '呼吸', 'shortcut': 'Alt+B'},
            'laugh_burst': {'tag': '[laughter]', 'name': '笑声爆发', 'shortcut': 'Alt+Shift+L'},
            'endofprompt': {'tag': '<|endofprompt|>', 'name': '指令结束符', 'shortcut': 'Alt+E'},
        }
    
    def set_voice_configs(self, configs: Dict[str, VoiceConfig]):
        self.voice_configs = configs
    
    def keyPressEvent(self, event):
        # Ctrl+数字：应用语音配置
        if event.modifiers() == Qt.ControlModifier:
            key = event.key()
            if Qt.Key_1 <= key <= Qt.Key_9:
                mode_index = key - Qt.Key_1
                if self.textCursor().hasSelection():
                    config_list = list(self.voice_configs.values())
                    if mode_index < len(config_list):
                        config_name = config_list[mode_index].name
                        self.apply_voice_config(config_name)
                return
        
        # Alt+快捷键：插入控制标签
        if event.modifiers() == Qt.AltModifier:
            key = event.key()
            if key == Qt.Key_S:  # Alt+S: 强调
                self.insert_tag('strong')
                return
            elif key == Qt.Key_L:  # Alt+L: 笑声
                self.insert_tag('laughter')
                return
            elif key == Qt.Key_B:  # Alt+B: 呼吸
                self.insert_tag('breath')
                return
            elif key == Qt.Key_E:  # Alt+E: 指令结束符
                self.insert_tag('endofprompt')
                return
        
        # Alt+Shift+L: 笑声爆发
        if event.modifiers() == (Qt.AltModifier | Qt.ShiftModifier):
            if event.key() == Qt.Key_L:
                self.insert_tag('laugh_burst')
                return
        
        super().keyPressEvent(event)
    
    def show_context_menu(self, position: QPoint):
        menu = QMenu(self)
        
        if self.textCursor().hasSelection():
            copy_action = QAction("复制", self)
            copy_action.triggered.connect(self.copy)
            menu.addAction(copy_action)
            
            cut_action = QAction("剪切", self)
            cut_action.triggered.connect(self.cut)
            menu.addAction(cut_action)
        
        paste_action = QAction("粘贴", self)
        paste_action.triggered.connect(self.paste)
        menu.addAction(paste_action)
        
        menu.addSeparator()
        
        select_all_action = QAction("全选", self)
        select_all_action.triggered.connect(self.selectAll)
        menu.addAction(select_all_action)
        
        # 快捷指令菜单
        if self.textCursor().hasSelection() or True:
            menu.addSeparator()
            tag_menu = menu.addMenu("🏷️ 插入控制标签")
            
            for tag_key, tag_info in self.quick_tags.items():
                action = QAction(f"{tag_info['name']} ({tag_info['shortcut']})", self)
                action.triggered.connect(lambda checked, key=tag_key: self.insert_tag(key))
                tag_menu.addAction(action)
        
        # 语音配置菜单
        if self.textCursor().hasSelection() and self.voice_configs:
            menu.addSeparator()
            voice_menu = menu.addMenu("🎤 应用语音配置")
            
            for i, (config_name, config) in enumerate(self.voice_configs.items()):
                action = QAction(f"Ctrl+{i+1}: {config_name} ({config.mode})", self)
                action.triggered.connect(lambda checked, name=config_name: self.apply_voice_config(name))
                voice_menu.addAction(action)
        
        menu.exec_(self.mapToGlobal(position))
    
    def insert_tag(self, tag_key: str):
        """插入控制标签"""
        if tag_key not in self.quick_tags:
            return
        
        tag_info = self.quick_tags[tag_key]
        cursor = self.textCursor()
        
        if cursor.hasSelection():
            # 选中文本时，用标签包裹
            selected_text = cursor.selectedText()
            if '{}' in tag_info['tag']:
                new_text = tag_info['tag'].format(selected_text)
            else:
                new_text = tag_info['tag'] + selected_text
            cursor.insertText(new_text)
        else:
            # 未选中时，直接插入标签
            if '{}' in tag_info['tag']:
                # 有占位符的标签，插入后选中中间部分
                tag_parts = tag_info['tag'].split('{}')
                cursor.insertText(tag_parts[0])
                start_pos = cursor.position()
                cursor.insertText(tag_parts[1])
                cursor.setPosition(start_pos)
            else:
                cursor.insertText(tag_info['tag'])
    
    def apply_voice_config(self, config_name: str):
        if config_name not in self.voice_configs:
            return
        
        config = self.voice_configs[config_name]
        cursor = self.textCursor()
        
        if cursor.hasSelection():
            char_format = QTextCharFormat()
            char_format.setBackground(QColor(config.color))
            char_format.setProperty(QTextCharFormat.UserProperty, config_name)
            cursor.mergeCharFormat(char_format)
    
    def get_text_segments(self) -> List[Tuple[str, VoiceConfig]]:
        """提取按颜色分段的文本段落"""
        segments = []
        document = self.document()
        full_text = self.toPlainText()
        
        if not full_text.strip():
            return segments
        
        current_segment = ""
        current_config = None
        
        for i in range(len(full_text)):
            cursor = QTextCursor(document)
            cursor.setPosition(i)
            cursor.setPosition(i + 1, QTextCursor.KeepAnchor)
            
            char = full_text[i]
            char_format = cursor.charFormat()
            config_name = char_format.property(QTextCharFormat.UserProperty)
            
            if config_name and config_name in self.voice_configs:
                char_config = self.voice_configs[config_name]
            else:
                if self.voice_configs:
                    char_config = list(self.voice_configs.values())[0]
                else:
                    continue
            
            if current_config is not None and (
                current_config.name != char_config.name or 
                char == '\n'
            ):
                if current_segment.strip():
                    segments.append((current_segment.strip(), current_config))
                current_segment = ""
                current_config = None
            
            if char == '\n':
                continue
                
            if char.strip():
                if current_config is None:
                    current_config = char_config
                current_segment += char
            elif current_segment:
                current_segment += char
        
        if current_segment.strip() and current_config:
            segments.append((current_segment.strip(), current_config))
        
        return segments


# ==================== 音频生成线程 ====================

class AudioGenerationWorker(QThread):
    """音频生成工作线程"""
    progress = pyqtSignal(str)  # 日志消息
    finished = pyqtSignal(list)  # 生成的文件列表
    error = pyqtSignal(str)  # 错误消息
    segment_finished = pyqtSignal(int, list)  # 段落索引, 生成的文件列表
    
    def __init__(self, segments: List[TaskSegment], output_dir: str, 
                 project_name: str, cosyvoice_model=None):
        super().__init__()
        self.segments = segments
        self.output_dir = output_dir
        self.project_name = project_name
        self.cosyvoice = cosyvoice_model
        self.is_running = True
    
    def stop(self):
        self.is_running = False
    
    def run(self):
        try:
            # 如果没有模型，先加载
            if self.cosyvoice is None:
                self.progress.emit("📦 正在加载CosyVoice模型...")
                self.cosyvoice = self.load_model()
                self.progress.emit("✅ 模型加载成功")
            
            # 导入必要的模块
            from cosyvoice.utils.file_utils import load_wav
            import torchaudio
            
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            all_generated_files = []
            
            # 按段落生成
            for segment in self.segments:
                if not self.is_running:
                    break
                
                # 设置随机种子
                import torch
                import random
                torch.manual_seed(segment.seed)
                random.seed(segment.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(segment.seed)
                    torch.cuda.manual_seed_all(segment.seed)
                
                self.progress.emit(f"🎵 正在生成第 {segment.index} 段...")
                self.progress.emit(f"   文本: {segment.text}")
                self.progress.emit(f"   配置: {segment.voice_config.name} ({segment.mode})")
                self.progress.emit(f"   种子: {segment.seed}")
                
                # 加载参考音频
                if not segment.voice_config.prompt_audio or not os.path.exists(segment.voice_config.prompt_audio):
                    self.progress.emit(f"⚠️ 参考音频不存在，跳过")
                    continue
                
                prompt_speech_16k = load_wav(segment.voice_config.prompt_audio, 16000)
                
                # 生成音频 - 同一次运行的所有片段作为一个版本
                segment_files = []
                
                inference_func = self.get_inference_function(segment)
                
                for sub_idx, result in enumerate(inference_func(segment, prompt_speech_16k)):
                    if not self.is_running:
                        break
                    
                    # 生成文件名：使用run_count+1作为版本号
                    filename = self.generate_filename(segment, sub_idx, segment.run_count + 1)
                    filepath = os.path.join(self.output_dir, filename)
                    
                    # 保存音频
                    torchaudio.save(filepath, result['tts_speech'], self.cosyvoice.sample_rate)
                    segment_files.append(filepath)
                    all_generated_files.append(filepath)
                    
                    self.progress.emit(f"✅ 保存: {filename}")
                
                # 将这一批文件作为新版本添加
                if segment_files:
                    segment.add_version(segment_files)
                    self.progress.emit(f"📦 版本 v{segment.run_count} 包含 {len(segment_files)} 个片段")
                
                # 发送段落完成信号
                self.segment_finished.emit(segment.index, segment_files)
            
            if self.is_running:
                self.finished.emit(all_generated_files)
            
        except Exception as e:
            self.error.emit(f"生成失败: {str(e)}")
    
    def load_model(self):
        """加载CosyVoice模型"""
        model_dir = 'pretrained_models/CosyVoice2-0.5B'
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        
        sys.path.append('third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import CosyVoice2
        
        return CosyVoice2(
            model_dir, 
            load_jit=False, 
            load_trt=False, 
            load_vllm=False, 
            fp16=False
        )
    
    def get_inference_function(self, segment: TaskSegment):
        """获取推理函数"""
        if segment.mode == '零样本复制':
            def inference(seg, prompt_audio):
                return self.cosyvoice.inference_zero_shot(
                    seg.text, seg.voice_config.prompt_text, 
                    prompt_audio, stream=False
                )
            return inference
        
        elif segment.mode == '精细控制':
            def inference(seg, prompt_audio):
                return self.cosyvoice.inference_cross_lingual(
                    seg.text, prompt_audio, stream=False
                )
            return inference
        
        elif segment.mode == '指令控制':
            def inference(seg, prompt_audio):
                return self.cosyvoice.inference_instruct2(
                    seg.text, seg.instruct_text, 
                    prompt_audio, stream=False
                )
            return inference
        
        else:  # 流式输入
            def inference(seg, prompt_audio):
                return self.cosyvoice.inference_zero_shot(
                    seg.text, seg.voice_config.prompt_text, 
                    prompt_audio, stream=True
                )
            return inference
    
    def generate_filename(self, segment: TaskSegment, sub_index: int, version: int) -> str:
        """生成文件名: 段落序号_版本号_文本预览_片段序号.wav"""
        # 文本预览（10个字符）
        text_preview = self.sanitize_filename(segment.text[:10])
        
        # 格式：段落_版本_文本_片段.wav
        # 只有一个片段时不显示片段号
        return f"{segment.index}_{version}_{text_preview}_{sub_index+1}.wav"
    
    def sanitize_filename(self, text: str) -> str:
        """处理文件名，符合Windows规则"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            text = text.replace(char, '')
        text = ''.join(char for char in text if ord(char) >= 32)
        text = text.replace(' ', '_').replace('\n', '_').replace('\t', '_')
        while '__' in text:
            text = text.replace('__', '_')
        text = text.strip('_')
        return text or 'audio'


# ==================== 界面1: 文本编辑 ====================

class TextEditInterface(QWidget):
    """文本编辑界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        title = SubtitleLabel("📝 文本编辑")
        layout.addWidget(title)
        
        # 文本编辑区
        text_card = SimpleCardWidget()
        text_layout = QVBoxLayout(text_card)
        
        text_label = BodyLabel("输入文本内容 (右键或Ctrl+数字应用语音模式):")
        text_layout.addWidget(text_label)
        
        self.text_edit = CustomTextEdit()
        self.text_edit.setPlaceholderText(
            "请输入要转换为语音的文本内容...\n\n"
            "使用右键菜单或Ctrl+数字快捷键为选中的文本应用不同的语音模式。\n"
            "不同颜色代表不同的语音配置。"
        )
        text_layout.addWidget(self.text_edit)
        
        layout.addWidget(text_card)
        
        # 按钮区
        button_layout = QHBoxLayout()
        
        self.quick_run_button = PrimaryPushButton("⚡ 一键运行")
        self.quick_run_button.setToolTip("直接按顺序推理生成所有音频")
        button_layout.addWidget(self.quick_run_button)
        
        self.to_task_button = PushButton("📋 转成计划任务")
        self.to_task_button.setToolTip("转到任务计划界面，可以单独调整每段的参数")
        button_layout.addWidget(self.to_task_button)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def set_voice_configs(self, configs: Dict[str, VoiceConfig]):
        self.text_edit.set_voice_configs(configs)
    
    def get_text_segments(self) -> List[Tuple[str, VoiceConfig]]:
        return self.text_edit.get_text_segments()


# ==================== 界面2: 任务计划 ====================

class TaskPlanInterface(QWidget):
    """任务计划界面"""
    
    run_single_segment = pyqtSignal(int)  # 运行单个段落
    run_all_segments = pyqtSignal()  # 运行全部段落
    merge_audio = pyqtSignal()  # 合成音频
    play_audio = pyqtSignal(str)  # 播放音频
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_segments: List[TaskSegment] = []
        self.output_dir = "./output"
        self.project_name = "project"
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 标题和设置
        header_layout = QHBoxLayout()
        
        title = SubtitleLabel("📋 任务计划")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # 输出设置
        output_label = BodyLabel("输出目录:")
        header_layout.addWidget(output_label)
        
        self.output_edit = LineEdit()
        self.output_edit.setText(self.output_dir)
        self.output_edit.setFixedWidth(200)
        self.output_edit.textChanged.connect(self.on_output_changed)
        header_layout.addWidget(self.output_edit)
        
        browse_button = ToolButton()
        browse_button.setIcon(FluentIcon.FOLDER)
        browse_button.clicked.connect(self.browse_output_dir)
        header_layout.addWidget(browse_button)
        
        # 项目名称
        project_label = BodyLabel("项目名:")
        header_layout.addWidget(project_label)
        
        self.project_edit = LineEdit()
        self.project_edit.setText(self.project_name)
        self.project_edit.setFixedWidth(150)
        self.project_edit.textChanged.connect(self.on_project_changed)
        header_layout.addWidget(self.project_edit)
        
        layout.addLayout(header_layout)
        
        # 任务表格
        self.table = TableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "段落", "内容", "音色", "模式", "指令文本", "种子", "运行", "音频", "播放"
        ])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Fixed)  # 种子
        header.setSectionResizeMode(6, QHeaderView.Fixed)  # 运行
        header.setSectionResizeMode(7, QHeaderView.Fixed)  # 音频
        header.setSectionResizeMode(8, QHeaderView.Fixed)  # 播放
        
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(2, 120)
        self.table.setColumnWidth(3, 120)
        self.table.setColumnWidth(5, 80)   # 种子列
        self.table.setColumnWidth(6, 80)   # 运行按钮
        self.table.setColumnWidth(7, 200)  # 音频选择列
        self.table.setColumnWidth(8, 70)   # 播放按钮列
        
        layout.addWidget(self.table)
        
        # 底部按钮
        bottom_layout = QHBoxLayout()
        
        self.run_all_button = PrimaryPushButton("▶️ 全部运行")
        self.run_all_button.clicked.connect(self.run_all_segments.emit)
        bottom_layout.addWidget(self.run_all_button)
        
        self.merge_button = PushButton("🔧 合成音频")
        self.merge_button.clicked.connect(self.merge_audio.emit)
        bottom_layout.addWidget(self.merge_button)
        
        bottom_layout.addStretch()
        
        # 日志
        self.log_text = PlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setPlaceholderText("任务执行日志...")
        
        layout.addWidget(self.log_text)
        layout.addLayout(bottom_layout)
    
    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir)
        if directory:
            self.output_edit.setText(directory)
    
    def on_output_changed(self, text: str):
        self.output_dir = text
    
    def on_project_changed(self, text: str):
        self.project_name = text
    
    def load_segments(self, segments: List[Tuple[str, VoiceConfig]]):
        """加载文本段落到任务表格"""
        self.task_segments = [
            TaskSegment(i+1, text, config) 
            for i, (text, config) in enumerate(segments)
        ]
        self.update_table()
        self.add_log(f"✅ 已加载 {len(self.task_segments)} 个任务段落")
    
    def update_table(self):
        """更新任务表格"""
        self.table.setRowCount(len(self.task_segments))
        
        for i, segment in enumerate(self.task_segments):
            # 段落序号
            index_item = QTableWidgetItem(str(segment.index))
            index_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, index_item)
            
            # 内容
            content_item = QTableWidgetItem(segment.text[:50] + ('...' if len(segment.text) > 50 else ''))
            self.table.setItem(i, 1, content_item)
            
            # 音色
            voice_combo = ComboBox()
            voice_combo.addItem(segment.voice_config.name)
            voice_combo.setCurrentText(segment.voice_config.name)
            self.table.setCellWidget(i, 2, voice_combo)
            
            # 模式
            mode_combo = ComboBox()
            mode_combo.addItems(["零样本复制", "精细控制", "指令控制", "流式输入"])
            mode_combo.setCurrentText(segment.mode)
            mode_combo.currentTextChanged.connect(
                lambda text, idx=i: self.on_mode_changed(idx, text)
            )
            self.table.setCellWidget(i, 3, mode_combo)
            
            # 指令文本
            instruct_edit = LineEdit()
            instruct_edit.setText(segment.instruct_text)
            instruct_edit.textChanged.connect(
                lambda text, idx=i: self.on_instruct_changed(idx, text)
            )
            self.table.setCellWidget(i, 4, instruct_edit)
            
            # 随机种子
            seed_edit = LineEdit()
            seed_edit.setText(str(segment.seed))
            seed_edit.setPlaceholderText("42")
            seed_edit.textChanged.connect(
                lambda text, idx=i: self.on_seed_changed(idx, text)
            )
            self.table.setCellWidget(i, 5, seed_edit)
            
            # 运行按钮
            run_button = PushButton("▶️")
            run_button.setFixedWidth(60)
            run_button.clicked.connect(lambda checked, idx=i: self.run_single_segment.emit(idx))
            self.table.setCellWidget(i, 6, run_button)
            
            # 音频选择 - 显示版本_片段格式
            audio_combo = ComboBox()
            if segment.versions:
                options = segment.get_all_audio_options()
                for ver, seg, filepath in options:
                    # 显示格式：v版本号_片段号: 文件名
                    display_name = f"v{ver}_{seg}: {os.path.basename(filepath)}"
                    audio_combo.addItem(display_name)
                
                # 计算当前选中项的索引
                current_idx = 0
                for idx, (ver, seg, _) in enumerate(options):
                    if ver - 1 == segment.current_version and seg - 1 == segment.current_segment:
                        current_idx = idx
                        break
                audio_combo.setCurrentIndex(current_idx)
                
                # 存储options到combo的userData中
                for idx, (ver, seg, filepath) in enumerate(options):
                    audio_combo.setItemData(idx, (ver, seg))
                
                audio_combo.currentIndexChanged.connect(
                    lambda idx, seg_idx=i, cb=audio_combo: self.on_audio_combo_changed(seg_idx, idx, cb)
                )
            else:
                audio_combo.addItem("未生成")
            # 不设置固定宽度，让它自适应列宽
            self.table.setCellWidget(i, 7, audio_combo)
            
            # 播放按钮
            play_button = PushButton("🔊")
            play_button.setFixedWidth(55)
            play_button.setEnabled(bool(segment.current_audio))
            play_button.clicked.connect(
                lambda checked, idx=i: self.on_play_audio(idx)
            )
            self.table.setCellWidget(i, 8, play_button)
    
    def on_mode_changed(self, index: int, mode: str):
        if 0 <= index < len(self.task_segments):
            self.task_segments[index].mode = mode
    
    def on_instruct_changed(self, index: int, text: str):
        if 0 <= index < len(self.task_segments):
            self.task_segments[index].instruct_text = text
    
    def on_seed_changed(self, index: int, text: str):
        """随机种子改变事件"""
        if 0 <= index < len(self.task_segments):
            try:
                seed = int(text) if text.strip() else 42
                self.task_segments[index].seed = seed
            except ValueError:
                # 如果输入不是数字，保持原值
                pass
    
    def on_audio_combo_changed(self, seg_index: int, combo_index: int, combo_box):
        """音频选择框改变事件"""
        if 0 <= seg_index < len(self.task_segments):
            segment = self.task_segments[seg_index]
            # 从combo的userData获取版本和片段号
            version_segment = combo_box.itemData(combo_index)
            if version_segment:
                version, seg = version_segment
                if segment.set_audio(version, seg):
                    self.add_log(f"📻 切换到第 {segment.index} 段的 v{version}_{seg}")
    
    def on_audio_selected(self, index: int, filename: str):
        """保留兼容性"""
        if 0 <= index < len(self.task_segments):
            segment = self.task_segments[index]
            for file in segment.generated_files:
                if os.path.basename(file) == filename:
                    segment.current_audio = file
                    break
    
    def on_play_audio(self, index: int):
        if 0 <= index < len(self.task_segments):
            segment = self.task_segments[index]
            if segment.current_audio:
                self.play_audio.emit(segment.current_audio)
    
    def update_segment_audio(self, index: int, files: List[str]):
        """更新段落的音频文件列表"""
        for i, segment in enumerate(self.task_segments):
            if segment.index == index:
                # 重新创建下拉框
                audio_combo = ComboBox()
                if segment.versions:
                    options = segment.get_all_audio_options()
                    for ver, seg, filepath in options:
                        display_name = f"v{ver}_{seg}: {os.path.basename(filepath)}"
                        audio_combo.addItem(display_name)
                    
                    # 计算当前选中项的索引
                    current_idx = len(options) - 1  # 默认最新
                    audio_combo.setCurrentIndex(current_idx)
                    
                    # 存储options到combo的userData中
                    for idx, (ver, seg, filepath) in enumerate(options):
                        audio_combo.setItemData(idx, (ver, seg))
                    
                    audio_combo.currentIndexChanged.connect(
                        lambda idx, seg_idx=i, cb=audio_combo: self.on_audio_combo_changed(seg_idx, idx, cb)
                    )
                else:
                    audio_combo.addItem("未生成")
                self.table.setCellWidget(i, 7, audio_combo)
                
                # 启用播放按钮
                play_button = self.table.cellWidget(i, 8)
                if play_button:
                    play_button.setEnabled(True)
                
                break
    
    def add_log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{timestamp}] {message}")


# ==================== 界面3: 语音设置 ====================

class VoiceSettingsInterface(QWidget):
    """语音设置界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.voice_configs: List[VoiceConfig] = []
        self.config_dir = Path("./config")
        self.config_dir.mkdir(exist_ok=True)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 标题
        header_layout = QHBoxLayout()
        title = SubtitleLabel("🎙️ 语音设置")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # 模型管理按钮
        self.unload_model_button = PushButton("🗑️ 卸载模型")
        self.unload_model_button.setToolTip("从显存中卸载CosyVoice模型，释放资源")
        self.unload_model_button.clicked.connect(self.unload_model)
        header_layout.addWidget(self.unload_model_button)
        
        layout.addLayout(header_layout)
        
        # 配置表格
        self.table = TableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["名称", "模式", "参考文本", "参考音频", "指令文本", "颜色"])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Fixed)
        
        self.table.setColumnWidth(0, 120)
        self.table.setColumnWidth(1, 120)
        self.table.setColumnWidth(5, 80)
        
        layout.addWidget(self.table)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.add_button = PushButton("➕ 添加配置")
        self.add_button.clicked.connect(self.add_config)
        button_layout.addWidget(self.add_button)
        
        self.remove_button = PushButton("➖ 删除配置")
        self.remove_button.clicked.connect(self.remove_config)
        button_layout.addWidget(self.remove_button)
        
        button_layout.addStretch()
        
        self.load_button = PushButton("📂 加载配置")
        self.load_button.clicked.connect(self.load_config)
        button_layout.addWidget(self.load_button)
        
        self.save_button = PushButton("💾 保存配置")
        self.save_button.clicked.connect(self.save_config)
        button_layout.addWidget(self.save_button)
        
        self.apply_button = PrimaryPushButton("✅ 应用配置")
        self.apply_button.clicked.connect(self.apply_config)
        button_layout.addWidget(self.apply_button)
        
        layout.addLayout(button_layout)
        
        # 添加默认配置
        self.add_config()
    
    def unload_model(self):
        """卸载模型"""
        reply = QMessageBox.question(
            self, "确认卸载", 
            "确定要卸载CosyVoice模型吗？\n这将释放显存，但下次生成时需要重新加载。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 通知主窗口卸载模型
            main_window = self.window()
            if hasattr(main_window, 'unload_cosyvoice_model'):
                main_window.unload_cosyvoice_model()
                InfoBar.success(
                    title="卸载成功",
                    content="模型已从显存中卸载",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=2000,
                    parent=self
                )
    
    def add_config(self):
        config = VoiceConfig(
            name=f"语音配置{len(self.voice_configs) + 1}",
            mode="零样本复制",
            color=f"#{hash(f'config{len(self.voice_configs)}') % 0xFFFFFF:06x}"
        )
        self.voice_configs.append(config)
        self.update_table()
    
    def remove_config(self):
        current_row = self.table.currentRow()
        if 0 <= current_row < len(self.voice_configs):
            self.voice_configs.pop(current_row)
            self.update_table()
    
    def update_table(self):
        self.table.setRowCount(len(self.voice_configs))
        
        for i, config in enumerate(self.voice_configs):
            # 名称
            name_edit = LineEdit()
            name_edit.setText(config.name)
            name_edit.textChanged.connect(lambda text, idx=i: self.update_config_name(idx, text))
            self.table.setCellWidget(i, 0, name_edit)
            
            # 模式
            mode_combo = ComboBox()
            mode_combo.addItems(["零样本复制", "精细控制", "指令控制", "流式输入"])
            mode_combo.setCurrentText(config.mode)
            mode_combo.currentTextChanged.connect(lambda text, idx=i: self.update_config_mode(idx, text))
            self.table.setCellWidget(i, 1, mode_combo)
            
            # 参考文本
            prompt_text_edit = LineEdit()
            prompt_text_edit.setText(config.prompt_text)
            prompt_text_edit.textChanged.connect(lambda text, idx=i: self.update_config_prompt_text(idx, text))
            self.table.setCellWidget(i, 2, prompt_text_edit)
            
            # 参考音频
            audio_layout = QHBoxLayout()
            audio_edit = LineEdit()
            audio_edit.setText(config.prompt_audio)
            audio_edit.textChanged.connect(lambda text, idx=i: self.update_config_prompt_audio(idx, text))
            
            browse_button = ToolButton()
            browse_button.setIcon(FluentIcon.FOLDER)
            browse_button.clicked.connect(lambda checked, idx=i: self.browse_audio_file(idx))
            
            audio_widget = QWidget()
            audio_layout.addWidget(audio_edit)
            audio_layout.addWidget(browse_button)
            audio_layout.setContentsMargins(0, 0, 0, 0)
            audio_widget.setLayout(audio_layout)
            self.table.setCellWidget(i, 3, audio_widget)
            
            # 指令文本
            instruct_edit = LineEdit()
            instruct_edit.setText(config.instruct_text)
            instruct_edit.textChanged.connect(lambda text, idx=i: self.update_config_instruct_text(idx, text))
            self.table.setCellWidget(i, 4, instruct_edit)
            
            # 颜色
            color_button = PushButton()
            color_button.setStyleSheet(f"background-color: {config.color}; min-width: 60px; min-height: 30px;")
            color_button.clicked.connect(lambda checked, idx=i: self.choose_color(idx))
            self.table.setCellWidget(i, 5, color_button)
    
    def update_config_name(self, index: int, name: str):
        if 0 <= index < len(self.voice_configs):
            self.voice_configs[index].name = name
    
    def update_config_mode(self, index: int, mode: str):
        if 0 <= index < len(self.voice_configs):
            self.voice_configs[index].mode = mode
    
    def update_config_prompt_text(self, index: int, text: str):
        if 0 <= index < len(self.voice_configs):
            self.voice_configs[index].prompt_text = text
    
    def update_config_prompt_audio(self, index: int, audio: str):
        if 0 <= index < len(self.voice_configs):
            self.voice_configs[index].prompt_audio = audio
    
    def update_config_instruct_text(self, index: int, text: str):
        if 0 <= index < len(self.voice_configs):
            self.voice_configs[index].instruct_text = text
    
    def browse_audio_file(self, index: int):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", 
            "音频文件 (*.wav *.mp3 *.flac *.m4a);;所有文件 (*)"
        )
        if file_path and 0 <= index < len(self.voice_configs):
            self.voice_configs[index].prompt_audio = file_path
            self.update_table()
    
    def choose_color(self, index: int):
        if 0 <= index < len(self.voice_configs):
            color = QColorDialog.getColor(QColor(self.voice_configs[index].color), self)
            if color.isValid():
                self.voice_configs[index].color = color.name()
                self.update_table()
    
    def save_config(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存配置文件", str(self.config_dir / "voice_config.json"),
            "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            try:
                config_data = [config.to_dict() for config in self.voice_configs]
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                
                InfoBar.success(
                    title="保存成功",
                    content="配置已保存",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=2000,
                    parent=self
                )
            except Exception as e:
                InfoBar.error(
                    title="保存失败",
                    content=f"保存配置时发生错误: {str(e)}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
    
    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载配置文件", str(self.config_dir),
            "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                self.voice_configs = [VoiceConfig.from_dict(data) for data in config_data]
                self.update_table()
                
                InfoBar.success(
                    title="加载成功",
                    content="配置已加载",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=2000,
                    parent=self
                )
            except Exception as e:
                InfoBar.error(
                    title="加载失败",
                    content=f"加载配置时发生错误: {str(e)}",
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=3000,
                    parent=self
                )
    
    def apply_config(self):
        InfoBar.success(
            title="应用成功",
            content="语音配置已应用",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )
    
    def get_voice_configs(self) -> Dict[str, VoiceConfig]:
        return {config.name: config for config in self.voice_configs}


# ==================== 主窗口 ====================

class CosyVoiceProApp(FluentWindow):
    """主应用程序窗口"""
    
    def __init__(self):
        super().__init__()
        self.cosyvoice_model = None
        self.current_worker = None
        self.media_player = QMediaPlayer()
        self.init_window()
        self.init_navigation()
        self.connect_signals()
    
    def init_window(self):
        self.setWindowTitle("CosyVoice Desktop")
        self.resize(1400, 900)
        
        # 设置窗口图标
        icon_path = "./icon.ico"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
    
    def init_navigation(self):
        # 界面1: 文本编辑
        self.text_interface = TextEditInterface()
        self.text_interface.setObjectName("TextEditInterface")
        
        # 界面2: 任务计划
        self.task_interface = TaskPlanInterface()
        self.task_interface.setObjectName("TaskPlanInterface")
        
        # 界面3: 语音设置
        self.voice_interface = VoiceSettingsInterface()
        self.voice_interface.setObjectName("VoiceSettingsInterface")
        
        self.addSubInterface(
            self.text_interface, 
            FluentIcon.EDIT, 
            "文本编辑",
            NavigationItemPosition.TOP
        )
        
        self.addSubInterface(
            self.task_interface, 
            FluentIcon.CALENDAR, 
            "任务计划",
            NavigationItemPosition.TOP
        )
        
        self.addSubInterface(
            self.voice_interface, 
            FluentIcon.MICROPHONE, 
            "语音设置",
            NavigationItemPosition.TOP
        )
    
    def connect_signals(self):
        # 语音设置应用
        self.voice_interface.apply_button.clicked.connect(self.apply_voice_settings)
        
        # 文本编辑按钮
        self.text_interface.quick_run_button.clicked.connect(self.quick_run)
        self.text_interface.to_task_button.clicked.connect(self.to_task_plan)
        
        # 任务计划按钮
        self.task_interface.run_single_segment.connect(self.run_single_segment)
        self.task_interface.run_all_segments.connect(self.run_all_segments)
        self.task_interface.merge_audio.connect(self.merge_all_audio)
        self.task_interface.play_audio.connect(self.play_audio)
    
    def apply_voice_settings(self):
        """应用语音设置"""
        configs = self.voice_interface.get_voice_configs()
        self.text_interface.set_voice_configs(configs)
    
    def quick_run(self):
        """一键运行"""
        segments = self.text_interface.get_text_segments()
        if not segments:
            InfoBar.warning(
                title="无内容",
                content="请输入文本并应用语音模式",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            return
        
        # 创建任务段落
        task_segments = [
            TaskSegment(i+1, text, config) 
            for i, (text, config) in enumerate(segments)
        ]
        
        # 开始生成
        self.start_generation(task_segments)
    
    def to_task_plan(self):
        """转到任务计划"""
        segments = self.text_interface.get_text_segments()
        if not segments:
            InfoBar.warning(
                title="无内容",
                content="请输入文本并应用语音模式",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            return
        
        # 加载到任务计划
        self.task_interface.load_segments(segments)
        
        # 切换到任务计划界面
        self.switchTo(self.task_interface)
        
        InfoBar.success(
            title="转换成功",
            content=f"已加载 {len(segments)} 个任务段落",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )
    
    def run_single_segment(self, index: int):
        """运行单个段落"""
        segment = self.task_interface.task_segments[index]
        self.task_interface.add_log(f"🚀 开始生成第 {segment.index} 段...")
        self.start_generation([segment])
    
    def run_all_segments(self):
        """运行所有段落"""
        segments = self.task_interface.task_segments
        if not segments:
            InfoBar.warning(
                title="无任务",
                content="请先添加任务段落",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            return
        
        self.task_interface.add_log(f"🚀 开始生成全部 {len(segments)} 段...")
        self.start_generation(segments)
    
    def start_generation(self, segments: List[TaskSegment]):
        """开始音频生成"""
        if self.current_worker and self.current_worker.isRunning():
            InfoBar.warning(
                title="正在运行",
                content="已有任务正在运行中",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            return
        
        # 创建工作线程
        self.current_worker = AudioGenerationWorker(
            segments,
            self.task_interface.output_dir,
            self.task_interface.project_name,
            self.cosyvoice_model
        )
        
        # 连接信号
        self.current_worker.progress.connect(self.task_interface.add_log)
        self.current_worker.segment_finished.connect(self.task_interface.update_segment_audio)
        self.current_worker.finished.connect(self.on_generation_finished)
        self.current_worker.error.connect(self.on_generation_error)
        
        # 禁用按钮
        self.task_interface.run_all_button.setEnabled(False)
        
        # 启动线程
        self.current_worker.start()
    
    def on_generation_finished(self, files: List[str]):
        """生成完成"""
        self.task_interface.add_log(f"🎉 生成完成！共 {len(files)} 个文件")
        
        # 更新模型引用
        if self.current_worker:
            self.cosyvoice_model = self.current_worker.cosyvoice
        
        # 恢复按钮
        self.task_interface.run_all_button.setEnabled(True)
        
        InfoBar.success(
            title="生成完成",
            content=f"成功生成 {len(files)} 个音频文件",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=5000,
            parent=self
        )
    
    def on_generation_error(self, error: str):
        """生成错误"""
        self.task_interface.add_log(f"❌ {error}")
        self.task_interface.run_all_button.setEnabled(True)
        
        InfoBar.error(
            title="生成失败",
            content=error,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=5000,
            parent=self
        )
    
    def merge_all_audio(self):
        """合成所有音频 - 按版本合成所有片段"""
        segments = self.task_interface.task_segments
        files_to_merge = []
        
        for segment in segments:
            if not segment.versions:
                continue
            
            # 获取当前选中的版本号
            version_idx = segment.current_version
            
            # 获取该版本的所有片段并按顺序添加
            if 0 <= version_idx < len(segment.versions):
                version_files = segment.versions[version_idx]
                files_to_merge.extend(version_files)
                
                # 日志输出
                if len(version_files) > 1:
                    self.task_interface.add_log(
                        f"📦 段落{segment.index}: v{version_idx+1} ({len(version_files)}个片段)"
                    )
                else:
                    self.task_interface.add_log(
                        f"📦 段落{segment.index}: v{version_idx+1}"
                    )
        
        if not files_to_merge:
            InfoBar.warning(
                title="无音频",
                content="没有可合成的音频文件",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            return
        
        self.task_interface.add_log(f"🔧 开始合成 {len(files_to_merge)} 个音频片段...")
        
        # 合成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_file = self.merge_audio_files(
            files_to_merge, 
            self.task_interface.output_dir,
            f"{self.task_interface.project_name}_merged_{timestamp}.wav"
        )
        
        if merged_file:
            self.task_interface.add_log(f"✅ 合成完成: {os.path.basename(merged_file)}")
            InfoBar.success(
                title="合成完成",
                content=f"已保存到: {merged_file}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=5000,
                parent=self
            )
        else:
            self.task_interface.add_log("❌ 合成失败")
            InfoBar.error(
                title="合成失败",
                content="音频合成时发生错误",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
    
    def merge_audio_files(self, audio_files: List[str], output_dir: str, 
                         output_name: str) -> Optional[str]:
        """合并音频文件"""
        try:
            # 检查ffmpeg
            try:
                subprocess.run(['ffmpeg', '-version'], 
                             capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.task_interface.add_log("⚠️ 未找到ffmpeg")
                return None
            
            output_path = os.path.join(output_dir, output_name)
            
            # 创建文件列表
            filelist_path = os.path.join(output_dir, "filelist_temp.txt")
            with open(filelist_path, 'w', encoding='utf-8') as f:
                for audio_file in audio_files:
                    # Windows路径处理
                    abs_path = os.path.abspath(audio_file).replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")
            
            # 合并
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0',
                '-i', filelist_path,
                '-c', 'copy', '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 清理
            try:
                os.remove(filelist_path)
            except:
                pass
            
            return output_path if result.returncode == 0 else None
            
        except Exception as e:
            self.task_interface.add_log(f"❌ 合成错误: {str(e)}")
            return None
    
    def play_audio(self, filepath: str):
        """播放音频"""
        if not os.path.exists(filepath):
            InfoBar.warning(
                title="文件不存在",
                content="音频文件不存在",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            return
        
        url = QUrl.fromLocalFile(filepath)
        self.media_player.setMedia(QMediaContent(url))
        self.media_player.play()
        
        self.task_interface.add_log(f"🔊 播放: {os.path.basename(filepath)}")
    
    def unload_cosyvoice_model(self):
        """卸载CosyVoice模型"""
        if self.cosyvoice_model is not None:
            del self.cosyvoice_model
            self.cosyvoice_model = None
            
            # 清理缓存
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    app.setApplicationName("CosyVoice Desktop")
    app.setApplicationVersion("1.0")
    
    # 设置应用程序图标(任务栏图标)
    icon_path = "./icon.ico"
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    setTheme(Theme.AUTO)
    
    window = CosyVoiceProApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
