import sys
import io
import threading
import logging
import requests
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QScrollArea, QDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject, QTimer
from PyQt5.QtGui import QFont, QColor

from qfluentwidgets import (
    PushButton, PrimaryPushButton, SpinBox, SubtitleLabel, BodyLabel,
    FluentIcon, InfoBar, InfoBarPosition, CardWidget, CaptionLabel, TableWidget
)

from werkzeug.serving import make_server
from core import api

class APIDocDialog(QDialog):
    """API 文档对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("API 文档")
        self.setGeometry(100, 100, 600, 500)
        
        layout = QVBoxLayout(self)
        
        doc_text = QTextEdit()
        doc_text.setReadOnly(True)
        doc_text.setMarkdown("""# CosyVoice3 API 文档

## 1. 酒馆标准 TTS 端点
**方法:** POST  
**URL:** `http://127.0.0.1:9880/`

**请求体:**
```json
{
  "text": "要生成的文本",
  "speaker": "角色名称",
  "speed": 1.0
}
```

**返回:** WAV 音频文件

---

## 2. 获取角色列表
**方法:** GET  
**URL:** `http://127.0.0.1:9880/speakers`

**返回:**
```json
[
  {"name": "角色名", "voice_id": "角色名"},
  ...
]
```

---

## 3. 标准 API 端点
**方法:** POST  
**URL:** `http://127.0.0.1:9880/api/tts`

**请求体:**
```json
{
  "text": "要生成的文本",
  "character_name": "角色名称",
  "mode": "零样本复制|精细控制|指令控制",
  "speed": 1.0
}
```

**返回:** WAV 音频文件

---

## 4. 健康检查
**方法:** GET  
**URL:** `http://127.0.0.1:9880/api/health`

**返回:**
```json
{
  "status": "ok",
  "model": "CosyVoice3-0.5B",
  "characters": ["角色1", "角色2", ...]
}
```
""")
        layout.addWidget(doc_text)
        
        close_btn = PushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

class LogHandler(logging.Handler):
    """日志处理器，将日志发送到信号"""
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)

class StreamToSignal(object):
    """重定向 stdout/stderr 到信号"""
    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        self.signal.emit(text)

    def flush(self):
        pass

class RuntimeCharacterConfig:
    """运行时角色配置适配器"""
    def __init__(self, voice_settings_interface):
        self.voice_interface = voice_settings_interface

    def get_character(self, char_name: str) -> dict:
        """获取角色配置"""
        # 遍历 voice_interface 中的配置
        for config in self.voice_interface.voice_configs:
            if config.name == char_name:
                return config.to_dict()
        return None
    
    def list_characters(self) -> list:
        """列出所有角色"""
        return [config.name for config in self.voice_interface.voice_configs]

class APIServerThread(QThread):
    """API 服务线程"""
    log_signal = pyqtSignal(str)
    started_signal = pyqtSignal()
    stopped_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, host, port, model, config_manager):
        super().__init__()
        self.host = host
        self.port = port
        self.model = model
        self.config_manager = config_manager
        self.server = None
        self.is_running = False
        
        # 设置日志回调
        api.set_log_callback(self.on_api_log)

    def on_api_log(self, msg):
        """API 日志回调"""
        self.log_signal.emit(msg)

    def run(self):
        try:
            # 设置 API 全局变量
            api.set_globals(self.model, self.config_manager)
            
            # 创建服务器
            self.server = make_server(self.host, self.port, api.app)
            self.is_running = True
            self.started_signal.emit()
            self.log_signal.emit(f"🚀 API Server started at http://{self.host}:{self.port}")
            
            # 启动服务循环
            self.server.serve_forever()
            
        except Exception as e:
            self.error_signal.emit(str(e))
            self.log_signal.emit(f"❌ API Server error: {e}")
        finally:
            self.is_running = False
            self.stopped_signal.emit()

    def stop(self):
        if self.server:
            self.server.shutdown()

class APIPageInterface(QWidget):
    """API 服务管理界面"""
    
    log_received = pyqtSignal(str)
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.server_thread = None
        self.init_ui()
        self.connect_signals()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 左侧：控制面板
        left_panel = CardWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题和文档按钮
        title_layout = QHBoxLayout()
        title = SubtitleLabel("🔌 API 服务 (酒馆适配)")
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        doc_btn = PushButton("?")
        doc_btn.setMaximumWidth(35)
        doc_btn.clicked.connect(self.show_api_doc)
        title_layout.addWidget(doc_btn)
        
        left_layout.addLayout(title_layout)
        # 端口设置
        port_layout = QHBoxLayout()
        port_label = BodyLabel("端口:")
        self.port_spin = SpinBox(self)
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(9880)
        port_layout.addWidget(port_label)
        port_layout.addWidget(self.port_spin, 1)
        left_layout.addLayout(port_layout)
        
        # 角色列表部分
        char_title = SubtitleLabel("📋 角色列表")
        left_layout.addWidget(char_title)
        
        # 角色列表（使用TableWidget）
        self.character_table = TableWidget()
        self.character_table.setColumnCount(2)
        self.character_table.setHorizontalHeaderLabels(["角色名称", "推理模式"])
        self.character_table.setMaximumHeight(250)
        
        # 设置列宽
        header = self.character_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        
        left_layout.addWidget(self.character_table)
        
        # 手动刷新按钮
        refresh_btn = PushButton("🔄 刷新列表")
        refresh_btn.setIcon(FluentIcon.SYNC)
        refresh_btn.clicked.connect(self.refresh_character_list)
        left_layout.addWidget(refresh_btn)
        
        left_layout.addStretch()
        
        # 控制按钮
        self.start_btn = PrimaryPushButton("启动服务")
        self.start_btn.setIcon(FluentIcon.PLAY)
        self.start_btn.clicked.connect(self.toggle_server)
        left_layout.addWidget(self.start_btn)
        
        # 状态指示
        self.status_label = CaptionLabel("状态: 已停止")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label)
        
        layout.addWidget(left_panel, 1)
        
        # 右侧：日志输出
        right_panel = CardWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        
        log_title = SubtitleLabel("📝 运行日志")
        right_layout.addWidget(log_title)
        
        self.log_view = QTextEdit(self)
        self.log_view.setReadOnly(True)
        font = QFont("Courier New", 10)
        font.setFixedPitch(True)
        self.log_view.setFont(font)
        self.log_view.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0.03);
                color: #333333;
                border: 1px solid rgba(0, 0, 0, 0.08);
                border-radius: 6px;
                padding: 8px;
            }
        """)
        right_layout.addWidget(self.log_view)
        
        clear_btn = PushButton("清空日志")
        clear_btn.clicked.connect(self.log_view.clear)
        right_layout.addWidget(clear_btn)
        
        layout.addWidget(right_panel, 2)
        
        # 初始化角色列表（从本地加载）
        self.refresh_local_character_list()
    
    def refresh_local_character_list(self):
        """从本地配置加载角色列表"""
        if hasattr(self.main_window, 'voice_interface') and self.main_window.voice_interface:
            characters = []
            for config in self.main_window.voice_interface.voice_configs:
                characters.append({'name': config.name, 'mode': config.mode})
            self.update_character_list(characters)
    
    def show_api_doc(self):
        """显示 API 文档对话框"""
        dialog = APIDocDialog(self)
        dialog.exec_()

    def connect_signals(self):
        self.log_received.connect(self.append_log)

    def append_log(self, text):
        """添加日志到日志窗口，支持颜色"""
        if text.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_line = f"[{timestamp}] {text}"
            
            # 使用HTML格式来实现颜色
            color = "#333333"  # 默认黑色
            if "✅" in text or "成功" in text:
                color = "#00aa00"  # 绿色
            elif "❌" in text or "失败" in text or "异常" in text or "错误" in text:
                color = "#ff0000"  # 红色
            elif "⚠️" in text or "警告" in text:
                color = "#ff7700"  # 橙色
            elif "🔄" in text or "正在" in text:
                color = "#0077ff"  # 蓝色
            
            html_line = f'<span style="color: {color}">{log_line}</span>'
            self.log_view.append(html_line)
            
            # 滚动到底部
            cursor = self.log_view.textCursor()
            cursor.movePosition(cursor.End)
            self.log_view.setTextCursor(cursor)

    def toggle_server(self):
        if self.server_thread and self.server_thread.isRunning():
            # 停止服务
            self.start_btn.setEnabled(False)
            self.start_btn.setText("正在停止...")
            self.server_thread.stop()
            # 线程结束信号会处理 UI 更新
        else:
            # 启动服务
            # 检查模型是否加载
            if self.main_window.cosyvoice_model is None:
                # 尝试从 worker 加载，或者提示用户
                # 这里简单检查 main_window.cosyvoice_model
                # 如果没有，尝试触发加载（这比较复杂，因为加载是耗时的）
                # 暂时提示用户先去生成一次音频或手动加载
                
                # 检查是否有正在运行的 worker
                if self.main_window.current_worker and self.main_window.current_worker.cosyvoice:
                    self.main_window.cosyvoice_model = self.main_window.current_worker.cosyvoice
                
                if self.main_window.cosyvoice_model is None:
                    InfoBar.error(
                        title='模型未加载',
                        content="请先在主界面进行一次语音生成以加载模型，或等待模型初始化完成。",
                        orient=Qt.Horizontal,
                        isClosable=True,
                        position=InfoBarPosition.TOP,
                        duration=3000,
                        parent=self
                    )
                    return

            port = self.port_spin.value()
            
            # 创建运行时配置适配器
            runtime_config = RuntimeCharacterConfig(self.main_window.voice_interface)
            
            self.server_thread = APIServerThread(
                host="0.0.0.0",
                port=port,
                model=self.main_window.cosyvoice_model,
                config_manager=runtime_config
            )
            
            self.server_thread.log_signal.connect(self.log_received)
            self.server_thread.started_signal.connect(self.on_server_started)
            self.server_thread.stopped_signal.connect(self.on_server_stopped)
            self.server_thread.error_signal.connect(self.on_server_error)
            
            self.start_btn.setEnabled(False)
            self.start_btn.setText("正在启动...")
            self.server_thread.start()

    def on_server_started(self):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("停止服务")
        self.start_btn.setIcon(FluentIcon.PAUSE)
        self.status_label.setText("状态: 🟢 运行中")
        
        # 5秒后自动刷新一次角色列表
        auto_refresh_timer = QTimer(self)
        auto_refresh_timer.setSingleShot(True)
        auto_refresh_timer.timeout.connect(self.refresh_character_list)
        auto_refresh_timer.start(5000)  # 5秒后触发
        self.port_spin.setEnabled(False)
        
        InfoBar.success(
            title='服务已启动',
            content=f"API 服务正在运行于端口 {self.port_spin.value()}",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )

    def on_server_stopped(self):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("启动服务")
        self.start_btn.setIcon(FluentIcon.PLAY)
        self.status_label.setText("状态: 🔴 已停止")
        self.port_spin.setEnabled(True)
        self.log_received.emit("API Server stopped.")

    def on_server_error(self, error_msg):
        self.start_btn.setEnabled(True)
        self.start_btn.setText("启动服务")
        self.start_btn.setIcon(FluentIcon.PLAY)
        self.status_label.setText("状态: ⚠️ 错误")
        self.port_spin.setEnabled(True)
        
        InfoBar.error(
            title='服务错误',
            content=error_msg,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=5000,
            parent=self
        )
    
    def refresh_character_list(self):
        """从 API 刷新角色列表"""
        # 检查服务器线程是否存在且在运行
        if not self.server_thread:
            self.log_received.emit("⚠️ 服务线程未初始化")
            return
        
        if not self.server_thread.is_running:
            self.log_received.emit("⚠️ 服务未运行，无法刷新角色列表")
            return
        
        try:
            port = self.port_spin.value()
            url = f"http://127.0.0.1:{port}/speakers"
            self.log_received.emit(f"🔄 正在获取角色列表...")
            
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                characters = response.json()
                self.log_received.emit(f"✅ 获取成功，共 {len(characters)} 个角色")
                self.update_character_list(characters)
            else:
                self.log_received.emit(f"⚠️ 获取失败，状态码: {response.status_code}")
        except Exception as e:
            self.log_received.emit(f"❌ 获取角色列表异常: {str(e)}")
    
    def update_character_list(self, characters):
        """更新角色列表"""
        self.character_table.setRowCount(len(characters))
        
        try:
            # 获取所有角色配置
            voice_configs = self.main_window.voice_interface.get_voice_configs()
        except:
            voice_configs = {}
        
        for row, char in enumerate(characters):
            char_name = char.get('name', '')
            
            # 从配置获取模式
            if char_name in voice_configs:
                mode = voice_configs[char_name].mode if hasattr(voice_configs[char_name], 'mode') else '零样本复制'
            else:
                mode = '未知'
            
            # 创建表格项
            name_item = QTableWidgetItem(char_name)
            mode_item = QTableWidgetItem(mode)
            
            # 设置字体
            font = QFont('微软雅黑', 10)
            name_item.setFont(font)
            mode_item.setFont(font)
            
            # 添加到表格
            self.character_table.setItem(row, 0, name_item)
            self.character_table.setItem(row, 1, mode_item)
