import os
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar, QLineEdit

from qfluentwidgets import (
    SubtitleLabel,
    BodyLabel,
    CaptionLabel,
    ComboBox,
    LineEdit,
    PushButton,
    PrimaryPushButton,
    TextEdit,
    InfoBar,
    InfoBarPosition,
    CardWidget,
)

from core.config_manager import ConfigManager
from core.worker import ModelDownloadThread
from core.download import is_model_downloaded


class ModelDownloadInterface(QWidget):
    """模型下载页面（内置下载 + 路径设置 + 进度显示）"""

    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.download_thread = None
        self.init_ui()
        self.load_config()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(30, 30, 30, 30)

        title = SubtitleLabel("📥 模型下载与路径设置")
        layout.addWidget(title)

        tip = BodyLabel("在这里可直接下载模型、查看下载状态并单独设置路径，无需再用外部脚本。")
        tip.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(tip)

        status_card = CardWidget(self)
        status_layout = QHBoxLayout(status_card)
        status_layout.setContentsMargins(16, 12, 16, 12)
        status_layout.setSpacing(12)

        self.wetext_status_label = BodyLabel("WeText：⬜ 未下载")
        self.cosy_status_label = BodyLabel("CosyVoice3：⬜ 未下载")
        refresh_status_btn = PushButton("刷新状态")
        refresh_status_btn.clicked.connect(self.refresh_download_status)

        status_layout.addWidget(self.wetext_status_label)
        status_layout.addWidget(self.cosy_status_label)
        status_layout.addStretch()
        status_layout.addWidget(refresh_status_btn)
        layout.addWidget(status_card)

        path_card = CardWidget(self)
        path_card_layout = QVBoxLayout(path_card)
        path_card_layout.setContentsMargins(16, 14, 16, 14)
        path_card_layout.setSpacing(10)

        path_title = SubtitleLabel("📁 模型路径（独立设置）")
        path_card_layout.addWidget(path_title)

        wetext_row = QHBoxLayout()
        wetext_label = BodyLabel("WeText 路径")
        self.wetext_path_edit = LineEdit()
        self.wetext_path_edit.setPlaceholderText("例如：D:/Models/wetext")
        self.wetext_path_edit.textChanged.connect(self.on_wetext_path_changed)
        wetext_browse = PushButton("浏览")
        wetext_browse.clicked.connect(self.on_browse_wetext_path)
        wetext_row.addWidget(wetext_label)
        wetext_row.addWidget(self.wetext_path_edit, 1)
        wetext_row.addWidget(wetext_browse)
        path_card_layout.addLayout(wetext_row)

        cosy_row = QHBoxLayout()
        cosy_label = BodyLabel("CosyVoice3 路径")
        self.cosy_path_edit = LineEdit()
        self.cosy_path_edit.setPlaceholderText("例如：D:/Models/Fun-CosyVoice3-0.5B")
        self.cosy_path_edit.textChanged.connect(self.on_cosy_path_changed)
        cosy_browse = PushButton("浏览")
        cosy_browse.clicked.connect(self.on_browse_cosy_path)
        cosy_row.addWidget(cosy_label)
        cosy_row.addWidget(self.cosy_path_edit, 1)
        cosy_row.addWidget(cosy_browse)
        path_card_layout.addLayout(cosy_row)

        path_tip = CaptionLabel("提示：路径可不同盘符；程序会按各自路径判断是否已下载。")
        path_tip.setStyleSheet("color: gray;")
        path_card_layout.addWidget(path_tip)

        layout.addWidget(path_card)

        channel_layout = QHBoxLayout()
        channel_label = BodyLabel("下载渠道")
        self.channel_combo = ComboBox()
        self.channel_combo.addItems(["ModelScope（推荐）", "HuggingFace"])
        channel_layout.addWidget(channel_label)
        channel_layout.addStretch()
        channel_layout.addWidget(self.channel_combo)
        layout.addLayout(channel_layout)

        self.token_edit = LineEdit()
        self.token_edit.setPlaceholderText("可选：HuggingFace Token（仅 HuggingFace 时生效）")
        self.token_edit.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.token_edit)

        button_layout = QHBoxLayout()
        self.download_all_btn = PrimaryPushButton("一键下载默认模型")
        self.download_all_btn.clicked.connect(self.download_all_models)
        self.download_wetext_btn = PushButton("仅下载 WeText")
        self.download_wetext_btn.clicked.connect(self.download_wetext)
        self.download_cosy_btn = PushButton("仅下载 CosyVoice3")
        self.download_cosy_btn.clicked.connect(self.download_cosyvoice)
        button_layout.addWidget(self.download_all_btn)
        button_layout.addWidget(self.download_wetext_btn)
        button_layout.addWidget(self.download_cosy_btn)
        layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.progress_label = BodyLabel("进度：等待开始")
        layout.addWidget(self.progress_label)

        log_title = SubtitleLabel("📝 下载日志")
        layout.addWidget(log_title)

        self.log_view = TextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(260)
        layout.addWidget(self.log_view)

        clear_btn = PushButton("清空日志")
        clear_btn.clicked.connect(self.log_view.clear)
        layout.addWidget(clear_btn, alignment=Qt.AlignLeft)

        layout.addStretch()

    def load_config(self):
        default_wetext = "./pretrained_models/wetext"
        default_cosy = "./pretrained_models/Fun-CosyVoice3-0.5B"

        wetext_path = self.config_manager.get("wetext_model_path", default_wetext)
        cosy_path = self.config_manager.get("cosyvoice_model_path", default_cosy)

        self.wetext_path_edit.setText(wetext_path)
        self.cosy_path_edit.setText(cosy_path)
        self.refresh_download_status()

    def on_wetext_path_changed(self, path: str):
        if not path:
            return
        # 允许保存相对路径
        self.config_manager.set("wetext_model_path", path)
        self.refresh_download_status()

    def on_cosy_path_changed(self, path: str):
        if not path:
            return
        # 允许保存相对路径
        self.config_manager.set("cosyvoice_model_path", path)
        self.refresh_download_status()

    def on_browse_wetext_path(self):
        current = self.wetext_path_edit.text().strip() or "./pretrained_models"
        selected = QFileDialog.getExistingDirectory(self, "选择模型目录", os.path.abspath(current))
        if selected:
            self.wetext_path_edit.setText(selected)

    def on_browse_cosy_path(self):
        current = self.cosy_path_edit.text().strip() or "./pretrained_models"
        selected = QFileDialog.getExistingDirectory(self, "选择模型目录", os.path.abspath(current))
        if selected:
            self.cosy_path_edit.setText(selected)

    def get_model_paths(self):
        # 统一使用 core.download 中的逻辑来解析最终路径
        from core.download import get_model_catalog
        
        # 使用当前编辑框中的内容作为基础路径
        wetext_raw = self.wetext_path_edit.text().strip() or "./pretrained_models"
        cosy_raw = self.cosy_path_edit.text().strip() or "./pretrained_models"
        
        catalog = get_model_catalog("./pretrained_models", {
            "wetext": wetext_raw,
            "cosyvoice3": cosy_raw
        })
        
        return {
            "wetext": catalog["wetext"][3],
            "cosyvoice3": catalog["cosyvoice3"][3],
        }

    def refresh_download_status(self):
        model_paths = self.get_model_paths()
        wetext_ok = is_model_downloaded(model_paths["wetext"])
        cosy_ok = is_model_downloaded(model_paths["cosyvoice3"])

        if wetext_ok:
            self.wetext_status_label.setText("WeText：✅ 已下载")
            self.wetext_status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        else:
            self.wetext_status_label.setText("WeText：⬜ 未下载")
            self.wetext_status_label.setStyleSheet("color: #95a5a6;")

        if cosy_ok:
            self.cosy_status_label.setText("CosyVoice3：✅ 已下载")
            self.cosy_status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        else:
            self.cosy_status_label.setText("CosyVoice3：⬜ 未下载")
            self.cosy_status_label.setStyleSheet("color: #95a5a6;")

        self.config_manager.set("wetext_model_path", model_paths["wetext"])
        self.config_manager.set("cosyvoice_model_path", model_paths["cosyvoice3"])

    def get_download_method(self):
        return "huggingface" if self.channel_combo.currentIndex() == 1 else "modelscope"

    def get_token(self):
        token = self.token_edit.text().strip()
        return token if token else None

    def download_all_models(self):
        self.start_download(['wetext', 'cosyvoice3'])

    def download_wetext(self):
        self.start_download(['wetext'])

    def download_cosyvoice(self):
        self.start_download(['cosyvoice3'])

    def start_download(self, download_keys):
        if self.download_thread and self.download_thread.isRunning():
            self.show_warning("已有下载任务正在进行，请稍候。")
            return

        model_paths = self.get_model_paths()
        self.refresh_download_status()

        pending_keys = [key for key in download_keys if not is_model_downloaded(model_paths[key])]
        skipped_keys = [key for key in download_keys if key not in pending_keys]

        if skipped_keys:
            skipped_text = "、".join(["WeText" if key == "wetext" else "CosyVoice3" for key in skipped_keys])
            self.append_log(f"[{self.now()}] ⏭️ 已下载，跳过: {skipped_text}")

        if not pending_keys:
            self.show_warning("所选模型已下载完成，无需重复下载。")
            return

        self.progress_bar.setValue(0)
        self.progress_label.setText("进度：准备开始下载...")

        method = self.get_download_method()
        token = self.get_token()

        self.append_log(f"[{self.now()}] 🚀 开始下载，渠道: {method}")
        self.set_buttons_enabled(False)

        self.download_thread = ModelDownloadThread(
            download_method=method,
            token=token,
            download_keys=pending_keys,
            models_dir=None,
            model_paths=model_paths,
        )
        self.download_thread.progress.connect(self.on_download_progress)
        self.download_thread.log.connect(self.on_download_log)
        self.download_thread.finished.connect(self.on_download_finished)
        self.download_thread.error.connect(self.on_download_error)
        self.download_thread.start()

    def on_download_progress(self, value: int, status: str):
        self.progress_bar.setValue(max(0, min(100, value)))
        self.progress_label.setText(f"进度：{value}%  {status}")

    def on_download_log(self, message: str):
        self.append_log(f"[{self.now()}] {message}")

    def on_download_finished(self, result: dict):
        self.set_buttons_enabled(True)

        resolved_paths = result.get("resolved_paths", {})
        cosyvoice_path = resolved_paths.get("cosyvoice_model_path")
        wetext_path = resolved_paths.get("wetext_model_path")

        # 更新配置和界面上的路径输入框
        if cosyvoice_path:
            self.config_manager.set("cosyvoice_model_path", cosyvoice_path)
            self.cosy_path_edit.setText(cosyvoice_path)
        if wetext_path:
            self.config_manager.set("wetext_model_path", wetext_path)
            self.wetext_path_edit.setText(wetext_path)

        if result.get("all_success"):
            self.show_success("模型下载完成，路径已自动更新为模型实际存放位置。")
            self.progress_label.setText("进度：100% 下载完成")
        else:
            self.show_warning("部分模型下载失败，请查看日志并重试。")
            self.progress_label.setText("进度：任务结束（存在失败）")

        self.refresh_download_status()

    def on_download_error(self, error_msg: str):
        self.set_buttons_enabled(True)
        self.show_error(f"下载失败：{error_msg}")
        self.append_log(f"[{self.now()}] ❌ 下载线程错误: {error_msg}")

    def set_buttons_enabled(self, enabled: bool):
        self.download_all_btn.setEnabled(enabled)
        self.download_wetext_btn.setEnabled(enabled)
        self.download_cosy_btn.setEnabled(enabled)

    def append_log(self, text: str):
        self.log_view.append(text)

    @staticmethod
    def now() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def show_success(self, content: str):
        InfoBar.success(
            title='成功',
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self,
        )

    def show_warning(self, content: str):
        InfoBar.warning(
            title='提示',
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self,
        )

    def show_error(self, content: str):
        InfoBar.error(
            title='错误',
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=4000,
            parent=self,
        )
