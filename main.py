import sys
import os
import logging
import warnings

# 尝试屏蔽 QFluentWidgets 的 Pro 提示
try:
    from qfluentwidgets.common.config import ALERT
    import qfluentwidgets.common.config as qconfig
    qconfig.ALERT = ""
except ImportError:
    pass

# 过滤警告和日志
warnings.filterwarnings("ignore")

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon


class _FilteredStream:
    """过滤特定控制台提示，避免打扰用户。"""

    def __init__(self, stream, blocked_phrases):
        self._stream = stream
        self._blocked_phrases = blocked_phrases
        self._buffer = ""

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)

        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            full_line = line + "\n"
            if not any(phrase in full_line for phrase in self._blocked_phrases):
                self._stream.write(full_line)
        return len(data)

    def flush(self):
        if self._buffer:
            if not any(phrase in self._buffer for phrase in self._blocked_phrases):
                self._stream.write(self._buffer)
            self._buffer = ""
        self._stream.flush()

    def isatty(self):
        return self._stream.isatty() if hasattr(self._stream, 'isatty') else False

    def __getattr__(self, name):
        return getattr(self._stream, name)

def main():
    blocked_phrases = [
        "QFluentWidgets Pro is now released",
        "https://qfluentwidgets.com/pages/pro",
        "📢 Tips:",
        "qfluentwidgets.com/pages/pro",
    ]
    sys.stdout = _FilteredStream(sys.stdout, blocked_phrases)
    sys.stderr = _FilteredStream(sys.stderr, blocked_phrases)

    # 再次尝试在设置流之后 monkeypatch
    try:
        import qfluentwidgets.common.config as qconfig
        qconfig.ALERT = ""
    except:
        pass
    
    if hasattr(Qt, 'HighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QApplication(sys.argv)
    app.setApplicationName("CosyVoice Desktop")
    app.setApplicationVersion("1.0")
    
    icon_path = "./icon.ico"
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    print("QApplication 已初始化，正在加载 UI 组件...")

    try:
        from qfluentwidgets import setTheme, Theme
        from ui.main_window import CosyVoiceProApp
        
        setTheme(Theme.AUTO)
        
        window = CosyVoiceProApp()
        window.show()
        
        print("主窗口已显示，进入事件循环。")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"启动过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
