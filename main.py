import sys
import os
import logging
import warnings

# 过滤警告和日志
warnings.filterwarnings("ignore")

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

def main():
    print("正在启动 CosyVoice Desktop...")
    
    # High DPI scaling
    if hasattr(Qt, 'HighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    # 2. 创建 QApplication 实例
    # 必须在导入任何 UI 组件（如 qfluentwidgets 或自定义窗口）之前创建
    app = QApplication(sys.argv)
    app.setApplicationName("CosyVoice Desktop")
    app.setApplicationVersion("1.0")
    
    # 设置应用程序图标
    icon_path = "./icon.ico"
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    print("QApplication 已初始化，正在加载 UI 组件...")

    # 3. 延迟导入 UI 模块
    # 放在这里是为了确保 QApplication 已经存在
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
