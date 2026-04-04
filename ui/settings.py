import os

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QLineEdit
from PyQt5.QtCore import Qt
from qfluentwidgets import SwitchButton, SpinBox, SubtitleLabel, BodyLabel, PushButton, PrimaryPushButton, LineEdit, InfoBar, InfoBarPosition

from core.config_manager import ConfigManager

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
        title = SubtitleLabel("应用设置")
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

        save_llm_layout = QHBoxLayout()
        path_title = SubtitleLabel("路径设置")
        layout.addWidget(path_title)
        
        # 输出目录
        self.output_path_layout, self.output_path_edit = self.create_path_setting(
            "默认输出目录", "output_dir"
        )
        layout.addLayout(self.output_path_layout)
        
        # 提示：模型相关设置入口
        model_tip = BodyLabel("模型下载与模型路径设置已迁移到「模型下载」页面。")
        model_tip.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(model_tip)

        # AI 角色分配设置
        llm_title = SubtitleLabel("AI 角色分配")
        layout.addWidget(llm_title)

        self.llm_base_url_edit = self.create_line_setting(
            layout,
            "LLM Base URL",
            "llm_base_url",
            "例如: http://127.0.0.1:8000 或 http://127.0.0.1:8000/v1",
            auto_save=False
        )

        self.llm_api_key_edit = self.create_line_setting(
            layout,
            "LLM API Key",
            "llm_api_key",
            "可留空，兼容本地无鉴权服务",
            auto_save=False
        )
        self.llm_api_key_edit.setEchoMode(QLineEdit.Password)

        self.llm_model_edit = self.create_line_setting(
            layout,
            "LLM 模型名称",
            "llm_model",
            "例如: qwen/qwen3-32b",
            auto_save=False
        )

        timeout_layout = QHBoxLayout()
        timeout_label = BodyLabel("LLM 请求超时（秒）")
        self.llm_timeout_spin = SpinBox()
        self.llm_timeout_spin.setRange(5, 600)
        self.llm_timeout_spin.valueChanged.connect(
            lambda value: self.config_manager.set("llm_timeout_sec", value)
        )
        timeout_layout.addWidget(timeout_label)
        timeout_layout.addStretch()
        timeout_layout.addWidget(self.llm_timeout_spin)
        layout.addLayout(timeout_layout)

        auto_apply_layout = QHBoxLayout()
        auto_apply_label = BodyLabel("AI 分配后自动应用")
        self.llm_auto_apply_switch = SwitchButton()
        self.llm_auto_apply_switch.checkedChanged.connect(
            lambda checked: self.config_manager.set("llm_auto_apply", checked)
        )
        auto_apply_layout.addWidget(auto_apply_label)
        auto_apply_layout.addStretch()
        auto_apply_layout.addWidget(self.llm_auto_apply_switch)
        layout.addLayout(auto_apply_layout)

        self.default_speaker_edit = self.create_line_setting(
            layout,
            "默认角色名称",
            "default_speaker_name",
            "留空则回退到第一个角色，建议填写旁白角色",
            auto_save=False
        )

        llm_tip = BodyLabel(
            "文本页的“AI分配角色”会调用 OpenAI 兼容 Chat Completions 接口，"
            "结果会显示在文本页右侧的角色控制台；自动模式会直接写入角色标签。"
        )
        llm_tip.setWordWrap(True)
        llm_tip.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(llm_tip)

        layout.addStretch(1)

        save_llm_layout.addStretch()
        self.save_settings_button = PrimaryPushButton("保存设置")
        self.save_settings_button.clicked.connect(self.save_settings)
        save_llm_layout.addWidget(self.save_settings_button)
        layout.addLayout(save_llm_layout)

    def create_line_setting(self, layout, title, config_key, placeholder="", auto_save=True):
        container = QVBoxLayout()
        container.setSpacing(5)

        label = BodyLabel(title)
        container.addWidget(label)

        line_edit = LineEdit()
        line_edit.setPlaceholderText(placeholder)
        if auto_save:
            line_edit.textChanged.connect(lambda text: self.config_manager.set(config_key, text))
        container.addWidget(line_edit)

        layout.addLayout(container)
        return line_edit

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

        self.llm_base_url_edit.setText(self.config_manager.get("llm_base_url", ""))
        self.llm_api_key_edit.setText(self.config_manager.get("llm_api_key", ""))
        self.llm_model_edit.setText(self.config_manager.get("llm_model", ""))
        self.llm_timeout_spin.setValue(self.config_manager.get("llm_timeout_sec", 60))
        self.llm_auto_apply_switch.setChecked(self.config_manager.get("llm_auto_apply", False))
        self.default_speaker_edit.setText(self.config_manager.get("default_speaker_name", ""))
        self.output_path_edit.setText(self.config_manager.get("output_dir", "./output"))

    def save_settings(self):
        self.config_manager.config.update({
            "llm_base_url": self.llm_base_url_edit.text().strip(),
            "llm_api_key": self.llm_api_key_edit.text(),
            "llm_model": self.llm_model_edit.text().strip(),
            "llm_timeout_sec": self.llm_timeout_spin.value(),
            "llm_auto_apply": self.llm_auto_apply_switch.isChecked(),
            "default_speaker_name": self.default_speaker_edit.text().strip(),
            "output_dir": self.output_path_edit.text().strip(),
            "auto_load_model": self.auto_load_switch.isChecked(),
            "min_text_length": self.min_text_spin.value(),
        })
        self.config_manager.save_config()

        InfoBar.success(
            title="保存成功",
            content="设置已保存，下次启动会自动加载这些内容。",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2500,
            parent=self
        )

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
