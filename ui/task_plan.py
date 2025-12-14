import os
import datetime
from typing import List, Tuple, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QHeaderView, QTableWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from qfluentwidgets import (
    PushButton, PrimaryPushButton, TableWidget, LineEdit,
    ComboBox, FluentIcon, SubtitleLabel, BodyLabel, ToolButton, PlainTextEdit
)

from core.models import TaskSegment, VoiceConfig

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
            mode_combo.addItems(["零样本复刻", "精细控制", "指令控制"])
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
