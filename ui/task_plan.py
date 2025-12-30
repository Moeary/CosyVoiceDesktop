import os
import datetime
import json
from typing import List, Tuple, Optional, Dict

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QHeaderView, QTableWidgetItem,
    QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from qfluentwidgets import (
    PushButton, PrimaryPushButton, TableWidget, LineEdit,
    ComboBox, FluentIcon, SubtitleLabel, BodyLabel, ToolButton, PlainTextEdit,
    RoundMenu, Action, MessageBox
)

from core.config_manager import ConfigManager
from core.models import VoiceConfig, TaskSegment

class TaskPlanInterface(QWidget):
    """任务计划界面"""
    
    run_single_segment = pyqtSignal(int)  # 运行单个段落
    run_all_segments = pyqtSignal()  # 运行全部段落
    merge_audio = pyqtSignal()  # 合成音频
    play_audio = pyqtSignal(str)  # 播放音频
    
    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.task_segments: List[TaskSegment] = []
        self.all_voice_configs: Dict[str, VoiceConfig] = {}
        self.project_name = "project"
        self.init_ui()
    
    @property
    def output_dir(self):
        return self.config_manager.get("output_dir", "./output")

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 标题和设置
        header_layout = QHBoxLayout()
        
        title = SubtitleLabel("📋 任务计划")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # 项目名称
        project_label = BodyLabel("项目名:")
        header_layout.addWidget(project_label)
        
        self.project_edit = LineEdit()
        self.project_edit.setText(self.project_name)
        self.project_edit.setFixedWidth(150)
        self.project_edit.textChanged.connect(self.on_project_changed)
        header_layout.addWidget(self.project_edit)
        
        # 打开输出文件夹
        open_folder_button = ToolButton()
        open_folder_button.setIcon(FluentIcon.FOLDER_ADD)
        open_folder_button.setToolTip("打开输出文件夹")
        open_folder_button.clicked.connect(self.open_output_folder)
        header_layout.addWidget(open_folder_button)
        
        # 保存/加载按钮
        save_button = ToolButton()
        save_button.setIcon(FluentIcon.SAVE)
        save_button.setToolTip("保存计划")
        save_button.clicked.connect(self.save_plan)
        header_layout.addWidget(save_button)
        
        load_button = ToolButton()
        load_button.setIcon(FluentIcon.FOLDER)
        load_button.setToolTip("加载计划")
        load_button.clicked.connect(self.load_plan)
        header_layout.addWidget(load_button)
        
        layout.addLayout(header_layout)
        
        # 任务表格
        self.table = TableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "段落", "内容", "音色", "模式", "指令文本", "种子", "运行", "音频", "播放"
        ])
        
        # 启用右键菜单
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_table_context_menu)
        
        # 启用双击编辑
        self.table.cellDoubleClicked.connect(self.on_cell_double_clicked)
        # 监听内容修改
        self.table.itemChanged.connect(self.on_item_changed)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Interactive) # 允许调整
        header.setSectionResizeMode(5, QHeaderView.Fixed)  # 种子
        header.setSectionResizeMode(6, QHeaderView.Fixed)  # 运行
        header.setSectionResizeMode(7, QHeaderView.Interactive)  # 音频
        header.setSectionResizeMode(8, QHeaderView.Fixed)  # 播放
        
        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(2, 120)
        self.table.setColumnWidth(3, 120)
        self.table.setColumnWidth(4, 150)  # 默认宽度减小
        self.table.setColumnWidth(5, 60)   # 种子列
        self.table.setColumnWidth(6, 60)   # 运行按钮
        self.table.setColumnWidth(7, 150)  # 音频选择列减小
        self.table.setColumnWidth(8, 60)   # 播放按钮列
        
        # 隐藏默认的垂直表头（行号），因为我们已经有自定义的"段落"列
        self.table.verticalHeader().setVisible(False)
        
        layout.addWidget(self.table, 7) # 增加权重
        
        # 底部按钮
        bottom_layout = QHBoxLayout()
        
        self.run_all_button = PrimaryPushButton("▶️ 全部运行")
        self.run_all_button.clicked.connect(self.run_all_segments.emit)
        bottom_layout.addWidget(self.run_all_button)
        
        self.merge_button = PushButton("🔧 合成音频")
        self.merge_button.clicked.connect(self.merge_audio.emit)
        bottom_layout.addWidget(self.merge_button)

        self.add_row_button = PushButton("➕ 添加一行")
        self.add_row_button.setToolTip("在表格末尾添加一个新的空白行")
        self.add_row_button.clicked.connect(lambda: self.add_segment(len(self.task_segments)))
        bottom_layout.addWidget(self.add_row_button)
        
        bottom_layout.addStretch()
        
        # 日志
        self.log_text = PlainTextEdit()
        self.log_text.setReadOnly(True)
        # self.log_text.setMaximumHeight(100) # 移除固定高度
        self.log_text.setPlaceholderText("任务执行日志...")
        
        layout.addWidget(self.log_text, 3) # 增加权重，约占30%
        layout.addLayout(bottom_layout)
    
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
        self.table.blockSignals(True) # 阻止信号，防止触发itemChanged
        self.table.setRowCount(len(self.task_segments))
        
        for i, segment in enumerate(self.task_segments):
            # 段落序号
            index_item = QTableWidgetItem(str(segment.index))
            index_item.setTextAlignment(Qt.AlignCenter)
            index_item.setFlags(index_item.flags() & ~Qt.ItemIsEditable) # 序号不可编辑
            self.table.setItem(i, 0, index_item)
            
            # 内容 (可编辑)
            content_item = QTableWidgetItem(segment.text)
            content_item.setToolTip(segment.text) # 鼠标悬停显示全文
            self.table.setItem(i, 1, content_item)
            
            # 音色
            voice_combo = ComboBox()
            # 添加所有可用音色
            if self.all_voice_configs:
                for name in self.all_voice_configs.keys():
                    voice_combo.addItem(name)
            else:
                # 如果没有全局配置，至少添加当前的
                voice_combo.addItem(segment.voice_config.name)
            
            voice_combo.setCurrentText(segment.voice_config.name)
            voice_combo.currentTextChanged.connect(
                lambda text, idx=i: self.on_voice_changed(idx, text)
            )
            self.table.setCellWidget(i, 2, voice_combo)
            
            # 模式
            mode_combo = ComboBox()
            mode_combo.addItems(["零样本复制", "精细控制", "指令控制", "语音修补"])
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
        
        self.table.blockSignals(False)

    def on_voice_changed(self, index: int, voice_name: str):
        """音色改变事件"""
        if 0 <= index < len(self.task_segments) and voice_name in self.all_voice_configs:
            self.task_segments[index].voice_config = self.all_voice_configs[voice_name]
            # 自动更新模式为该音色的默认模式
            self.task_segments[index].mode = self.all_voice_configs[voice_name].mode
            # 刷新表格中的模式显示（可选，或者直接更新数据）
            # 这里为了简单，我们只更新数据，下次刷新表格时会显示
            # 如果需要即时更新UI，可以获取对应的ComboBox进行设置
            mode_combo = self.table.cellWidget(index, 3)
            if isinstance(mode_combo, ComboBox):
                mode_combo.setCurrentText(self.task_segments[index].mode)

    def on_item_changed(self, item):
        """表格内容改变事件"""
        row = item.row()
        col = item.column()
        if col == 1 and 0 <= row < len(self.task_segments): # 内容列
            self.task_segments[row].text = item.text()

    def on_cell_double_clicked(self, row, col):
        """双击单元格事件"""
        if col == 1 and 0 <= row < len(self.task_segments):
            # 弹出对话框编辑长文本
            from qfluentwidgets import MessageBoxBase, SubtitleLabel, TextEdit
            
            class TextEditDialog(MessageBoxBase):
                def __init__(self, text, parent=None):
                    super().__init__(parent)
                    self.titleLabel = SubtitleLabel("编辑文本内容", self)
                    self.textEdit = TextEdit(self)
                    self.textEdit.setPlainText(text)
                    self.textEdit.setMinimumHeight(200)
                    self.viewLayout.addWidget(self.titleLabel)
                    self.viewLayout.addWidget(self.textEdit)
                    self.widget.setMinimumWidth(500)
                    
            dialog = TextEditDialog(self.task_segments[row].text, self.window())
            if dialog.exec_():
                new_text = dialog.textEdit.toPlainText()
                self.task_segments[row].text = new_text
                self.table.item(row, 1).setText(new_text)

    def show_table_context_menu(self, pos):
        """显示表格右键菜单"""
        menu = RoundMenu(parent=self)
        
        # 获取当前选中的行
        selected_rows = sorted(set(item.row() for item in self.table.selectedItems()))
        current_row = self.table.currentRow()
        
        if selected_rows:
            menu.addAction(Action(FluentIcon.ADD, "在上方插入新行", self, triggered=lambda: self.add_segment(current_row)))
            menu.addAction(Action(FluentIcon.ADD, "在下方插入新行", self, triggered=lambda: self.add_segment(current_row + 1)))
            
            menu.addSeparator()
            menu.addAction(Action(FluentIcon.DELETE, "删除选中行", self, triggered=lambda: self.delete_segments(selected_rows)))
            
            if len(selected_rows) == 1:
                menu.addSeparator()
                menu.addAction(Action(FluentIcon.UP, "上移", self, triggered=lambda: self.move_segment(current_row, -1)))
                menu.addAction(Action(FluentIcon.DOWN, "下移", self, triggered=lambda: self.move_segment(current_row, 1)))
            
            menu.exec_(self.table.mapToGlobal(pos))

    def add_segment(self, index: int):
        """插入新段落"""
        if not self.all_voice_configs:
            MessageBox("提示", "请先在语音设置页面添加至少一个角色配置", self.window()).exec_()
            return
            
        # 使用第一个可用的配置作为默认
        default_config = list(self.all_voice_configs.values())[0]
        new_segment = TaskSegment(0, "请输入文本...", default_config)
        
        if 0 <= index <= len(self.task_segments):
            self.task_segments.insert(index, new_segment)
        else:
            self.task_segments.append(new_segment)
            
        self.renumber_segments()
        self.update_table()

    def delete_segments(self, rows: List[int]):
        """删除段落"""
        # 从后往前删，防止索引错位
        for row in sorted(rows, reverse=True):
            if 0 <= row < len(self.task_segments):
                self.task_segments.pop(row)
        
        self.renumber_segments()
        self.update_table()

    def move_segment(self, row: int, direction: int):
        """移动段落"""
        new_row = row + direction
        if 0 <= new_row < len(self.task_segments):
            self.task_segments[row], self.task_segments[new_row] = self.task_segments[new_row], self.task_segments[row]
            self.renumber_segments()
            self.update_table()
            self.table.selectRow(new_row)

    def renumber_segments(self):
        """重新编号"""
        for i, segment in enumerate(self.task_segments):
            segment.index = i + 1

    def open_output_folder(self):
        """打开输出文件夹"""
        path = os.path.abspath(os.path.join(self.output_dir, self.project_name))
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                self.add_log(f"❌ 创建目录失败: {e}")
                return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    
    def set_all_voice_configs(self, configs: Dict[str, VoiceConfig]):
        """设置所有可用的语音配置"""
        self.all_voice_configs = configs
        # 刷新表格以更新下拉框选项
        if self.task_segments:
            self.update_table()

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

    def save_plan(self):
        """保存任务计划"""
        # 默认保存路径为 output_dir/project_name
        default_dir = os.path.join(self.output_dir, self.project_name)
        if not os.path.exists(default_dir):
            try:
                os.makedirs(default_dir, exist_ok=True)
            except:
                default_dir = self.output_dir
        
        file_path, _ = QFileDialog.getSaveFileName(self, "保存任务计划", default_dir, "JSON Files (*.json)")
        if not file_path:
            return
            
        data = {
            "project_name": self.project_name,
            # "output_dir": self.output_dir, # 不再保存 output_dir，使用全局设置
            "segments": []
        }
        
        for segment in self.task_segments:
            seg_data = {
                "text": segment.text,
                "voice_config": segment.voice_config.to_dict(),
                "mode": segment.mode,
                "instruct_text": segment.instruct_text,
                "seed": segment.seed
            }
            data["segments"].append(seg_data)
            
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            self.add_log(f"💾 计划已保存至: {file_path}")
        except Exception as e:
            self.add_log(f"❌ 保存失败: {e}")

    def load_plan(self):
        """加载任务计划"""
        # 默认加载路径为 output_dir/project_name
        default_dir = os.path.join(self.output_dir, self.project_name)
        if not os.path.exists(default_dir):
             default_dir = self.output_dir
             
        file_path, _ = QFileDialog.getOpenFileName(self, "加载任务计划", default_dir, "JSON Files (*.json)")
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.project_name = data.get("project_name", "project")
            self.project_edit.setText(self.project_name)
            
            # self.output_dir = data.get("output_dir", "./output") # 不再加载 output_dir
            # self.output_edit.setText(self.output_dir)
            
            self.task_segments = []
            for i, seg_data in enumerate(data.get("segments", [])):
                voice_config_data = seg_data.get("voice_config", {})
                voice_config = VoiceConfig.from_dict(voice_config_data)
                
                # 如果全局配置中有同名的，优先使用全局配置（保持引用一致性），或者更新全局配置？
                # 这里我们直接使用保存的配置，但如果它在all_voice_configs中存在，最好关联上
                if self.all_voice_configs and voice_config.name in self.all_voice_configs:
                    # 检查内容是否一致，如果不一致可能需要警告或者覆盖
                    # 简单起见，我们信任保存的配置，但为了下拉框能正确显示，我们需要确保它在all_voice_configs中
                    # 或者我们只是使用它，下拉框会显示它的名字
                    pass
                
                segment = TaskSegment(
                    index=i+1,
                    text=seg_data.get("text", ""),
                    voice_config=voice_config,
                    mode=seg_data.get("mode"),
                    instruct_text=seg_data.get("instruct_text"),
                    seed=seg_data.get("seed", 42)
                )
                self.task_segments.append(segment)
            
            self.update_table()
            self.add_log(f"📂 已加载计划: {file_path}")
            
        except Exception as e:
            self.add_log(f"❌ 加载失败: {e}")
