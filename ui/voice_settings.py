import json
import os
from pathlib import Path
from typing import List, Dict

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QHeaderView, QMessageBox, QColorDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from qfluentwidgets import (
    PushButton, PrimaryPushButton, TableWidget, LineEdit,
    ComboBox, FluentIcon, SubtitleLabel, ToolButton, InfoBar, InfoBarPosition
)

from core.models import VoiceConfig
from core.config_manager import ConfigManager

class VoiceSettingsInterface(QWidget):
    """语音设置界面"""
    
    config_loaded = pyqtSignal()  # 配置加载完成信号
    
    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
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
        
        # 尝试加载默认配置
        default_config_path = self.config_dir / "config.json"
        if default_config_path.exists():
            self.load_config(str(default_config_path))
        else:
            # 添加默认配置
            self.add_config()
    
    # def unload_model(self):
    #     """卸载模型"""
    #     reply = QMessageBox.question(
    #         self, "确认卸载", 
    #         "确定要卸载CosyVoice模型吗？\n这将释放显存，但下次生成时需要重新加载。",
    #         QMessageBox.Yes | QMessageBox.No
    #     )
    #     
    #     if reply == QMessageBox.Yes:
    #         # 通知主窗口卸载模型
    #         main_window = self.window()
    #         if hasattr(main_window, 'unload_cosyvoice_model'):
    #             main_window.unload_cosyvoice_model()
    #             InfoBar.success(
    #                 title="卸载成功",
    #                 content="模型已从显存中卸载",
    #                 orient=Qt.Horizontal,
    #                 isClosable=True,
    #                 position=InfoBarPosition.TOP,
    #                 duration=2000,
    #                 parent=self
    #             )
    
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
            mode_combo.addItems(["零样本复制", "精细控制", "指令控制", "语音修补"])
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
    
    def save_config(self, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存配置文件", str(self.config_dir / "voice_config.json"),
                "JSON文件 (*.json);;所有文件 (*)"
            )
        
        if file_path:
            try:
                config_data = [config.to_dict() for config in self.voice_configs]
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                
                # 更新配置路径
                self.config_manager.set("voice_config_path", file_path)
                
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
    
    def load_config(self, file_path=None):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "加载配置文件", str(self.config_dir),
                "JSON文件 (*.json);;所有文件 (*)"
            )
        
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                self.voice_configs = [VoiceConfig.from_dict(data) for data in config_data]
                self.update_table()
                
                # 更新配置路径
                self.config_manager.set("voice_config_path", file_path)
                
                # 发送信号
                self.config_loaded.emit()
                
                InfoBar.success(
                    title="加载成功",
                    content="配置已加载并自动应用",
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
        # 自动保存到默认配置文件
        default_config_path = str(self.config_dir / "config.json")
        try:
            config_data = [config.to_dict() for config in self.voice_configs]
            with open(default_config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            # 更新配置路径
            self.config_manager.set("voice_config_path", default_config_path)
            
        except Exception as e:
            print(f"Auto-save failed: {e}")

        InfoBar.success(
            title="应用成功",
            content="语音配置已应用并保存",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )
    
    def get_voice_configs(self) -> Dict[str, VoiceConfig]:
        return {config.name: config for config in self.voice_configs}
