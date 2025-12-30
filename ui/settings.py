from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog
from qfluentwidgets import (
    ComboBox, SwitchButton, SpinBox, SubtitleLabel, BodyLabel, 
    setTheme, Theme, PushButton, ScrollArea, LineEdit
)
from core.config_manager import ConfigManager
import os

class SettingsInterface(QWidget):
    """设置界面 - 现在包含侧边栏内容"""
    
    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # 标题
        title = SubtitleLabel("⚙️ 应用设置")
        layout.addWidget(title)

        # 模型自动加载设置
        model_layout = QHBoxLayout()
        model_label = BodyLabel("启动时自动加载模型")
        self.auto_load_switch = SwitchButton()
        self.auto_load_switch.checkedChanged.connect(self.on_auto_load_changed)
        model_layout.addWidget(model_label)
        model_layout.addStretch()
        model_layout.addWidget(self.auto_load_switch)
        layout.addLayout(model_layout)
        
        # 提示信息
        model_tip = BodyLabel("启用后，应用启动时会自动加载 CosyVoice 模型（需要等待加载完成）")
        model_tip.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(model_tip)
        
        # 最小文本长度设置
        min_text_layout = QHBoxLayout()
        min_text_label = BodyLabel("最小推理文本长度")
        self.min_text_spin = SpinBox()
        self.min_text_spin.setMinimum(0)
        self.min_text_spin.setMaximum(10)
        self.min_text_spin.valueChanged.connect(self.on_min_text_changed)
        min_text_layout.addWidget(min_text_label)
        min_text_layout.addStretch()
        min_text_layout.addWidget(self.min_text_spin)
        layout.addLayout(min_text_layout)
        
        # 提示信息
        min_text_tip = BodyLabel("使用API时,低于此长度的文本会被跳过，避免推理失败（推荐4字符）")
        min_text_tip.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(min_text_tip)

        # 路径设置
        path_title = SubtitleLabel("📁 路径设置")
        layout.addWidget(path_title)
        
        # 输出目录
        self.output_path_layout, self.output_path_edit = self.create_path_setting(
            "默认输出目录", "output_dir"
        )
        layout.addLayout(self.output_path_layout)
        
        # CosyVoice 模型路径
        self.cosyvoice_path_layout, self.cosyvoice_path_edit = self.create_path_setting(
            "CosyVoice 模型路径", "cosyvoice_model_path"
        )
        layout.addLayout(self.cosyvoice_path_layout)
        
        # WeText 模型路径
        self.wetext_path_layout, self.wetext_path_edit = self.create_path_setting(
            "WeText 模型路径", "wetext_model_path"
        )
        layout.addLayout(self.wetext_path_layout)

        layout.addStretch()

    def create_path_setting(self, title, config_key, is_dir=True):
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        label = BodyLabel(title)
        layout.addWidget(label)
        
        input_layout = QHBoxLayout()
        line_edit = LineEdit()
        
        # Connect text change to save config
        line_edit.textChanged.connect(lambda text: self.config_manager.set(config_key, text))
        
        btn = PushButton("浏览")
        btn.clicked.connect(lambda: self.browse_path(line_edit, config_key, is_dir))
        
        input_layout.addWidget(line_edit)
        input_layout.addWidget(btn)
        layout.addLayout(input_layout)
        
        return layout, line_edit

    def browse_path(self, line_edit, config_key, is_dir):
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "选择目录", line_edit.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", line_edit.text())
            
        if path:
            # Convert to absolute path
            abs_path = os.path.abspath(path)
            line_edit.setText(abs_path)
            self.config_manager.set(config_key, abs_path)

    def load_settings(self):
        auto_load = self.config_manager.get("auto_load_model", False)
        self.auto_load_switch.setChecked(auto_load)
        
        min_text_length = self.config_manager.get("min_text_length", 5)
        self.min_text_spin.setValue(min_text_length)

        self.output_path_edit.setText(self.config_manager.get("output_dir", "./output"))
        self.cosyvoice_path_edit.setText(self.config_manager.get("cosyvoice_model_path", ""))
        self.wetext_path_edit.setText(self.config_manager.get("wetext_model_path", ""))

    def on_auto_load_changed(self, checked):
        self.config_manager.set("auto_load_model", checked)
    
    def on_min_text_changed(self, value):
        self.config_manager.set("min_text_length", value)
        # 实时更新API的最小文本长度
        try:
            from core import api
            api.set_min_text_length(value)
        except:
            pass
