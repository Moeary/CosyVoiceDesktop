from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from qfluentwidgets import (
    ComboBox, SwitchButton, Slider, SubtitleLabel, BodyLabel, 
    setTheme, Theme, PushButton
)
from core.config_manager import ConfigManager

class SettingsInterface(QWidget):
    """设置界面"""
    
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
        title = SubtitleLabel("⚙️ 设置")
        layout.addWidget(title)

        # 主题设置
        theme_layout = QHBoxLayout()
        theme_label = BodyLabel("主题模式")
        self.theme_combo = ComboBox()
        self.theme_combo.addItems(["Light", "Dark", "Auto"])
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        theme_layout.addWidget(theme_label)
        theme_layout.addStretch()
        theme_layout.addWidget(self.theme_combo)
        layout.addLayout(theme_layout)

        layout.addStretch()

    def load_settings(self):
        theme = self.config_manager.get("theme", "Light")
        self.theme_combo.setCurrentText(theme)

    def on_theme_changed(self, text):
        self.config_manager.set("theme", text)
        if text == "Light":
            setTheme(Theme.LIGHT)
        elif text == "Dark":
            setTheme(Theme.DARK)
        else:
            setTheme(Theme.AUTO)
