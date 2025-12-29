import sys
import os
import datetime
import gc
from typing import List, Optional

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QUrl, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from qfluentwidgets import (
    FluentWindow, FluentIcon, NavigationItemPosition, InfoBar, InfoBarPosition, setTheme, Theme,
    ComboBox, BodyLabel, PushButton
)

from core.models import TaskSegment
from core.worker import AudioGenerationWorker, ModelLoaderThread, ModelUnloaderThread
from core.utils import merge_audio_files
from core.config_manager import ConfigManager

from .text_edit import TextEditInterface
from .task_plan import TaskPlanInterface
from .voice_settings import VoiceSettingsInterface
from .settings import SettingsInterface
from .api_page import APIPageInterface

class CosyVoiceProApp(FluentWindow):
    """主应用程序窗口"""
    
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.cosyvoice_model = None
        self.current_worker = None
        self.model_loader_thread = None
        self.model_unloader_thread = None
        
        # Qt5 Audio Setup
        self.media_player = QMediaPlayer()
        # self.audio_output = QAudioOutput() # Qt5 doesn't need this for simple playback
        # self.media_player.setAudioOutput(self.audio_output)
        
        self.init_window()
        self.init_navigation()
        self.connect_signals()
        self.load_initial_config()
        
        # 在 GUI 加载完成后，检查是否需要加载模型
        QTimer.singleShot(500, self.load_model_if_enabled)
    
    def init_window(self):
        self.setWindowTitle("CosyVoice Desktop")
        self.resize(1400, 900)
        
        # 设置窗口图标
        icon_path = "./icon.ico"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        # 应用主题
        theme = self.config_manager.get("theme", "Light")
        if theme == "Light":
            setTheme(Theme.LIGHT)
        elif theme == "Dark":
            setTheme(Theme.DARK)
        else:
            setTheme(Theme.AUTO)
    
    def init_navigation(self):
        # 界面1: 文本编辑
        self.text_interface = TextEditInterface()
        self.text_interface.setObjectName("TextEditInterface")
        
        # 界面2: 任务计划
        self.task_interface = TaskPlanInterface()
        self.task_interface.setObjectName("TaskPlanInterface")
        
        # 界面3: 语音设置
        self.voice_interface = VoiceSettingsInterface(self.config_manager)
        self.voice_interface.setObjectName("VoiceSettingsInterface")
        
        # 界面4: 设置
        self.settings_interface = SettingsInterface(self.config_manager)
        self.settings_interface.setObjectName("SettingsInterface")
        
        # 界面5: API 服务
        self.api_interface = APIPageInterface(self)
        self.api_interface.setObjectName("APIPageInterface")
        
        self.addSubInterface(
            self.text_interface, 
            FluentIcon.EDIT, 
            "文本编辑",
            NavigationItemPosition.TOP
        )
        
        self.addSubInterface(
            self.task_interface, 
            FluentIcon.CALENDAR, 
            "任务计划",
            NavigationItemPosition.TOP
        )
        
        self.addSubInterface(
            self.voice_interface, 
            FluentIcon.MICROPHONE, 
            "语音设置",
            NavigationItemPosition.TOP
        )
        
        self.addSubInterface(
            self.api_interface, 
            FluentIcon.GLOBE, 
            "API 服务",
            NavigationItemPosition.TOP
        )
        
        # 在侧边栏添加模型加载按钮
        self.navigationInterface.addItem(
            routeKey='load_model',
            icon=FluentIcon.DOWNLOAD,
            text='加载模型',
            onClick=self.on_load_model_clicked,
            selectable=False,
            position=NavigationItemPosition.BOTTOM
        )

        # 在侧边栏添加模型卸载按钮
        self.navigationInterface.addItem(
            routeKey='unload_model',
            icon=FluentIcon.REMOVE,
            text='卸载模型',
            onClick=self.on_unload_model_clicked,
            selectable=False,
            position=NavigationItemPosition.BOTTOM
        )

        # 在侧边栏添加主题切换
        self.navigationInterface.addItem(
            routeKey='theme_toggle',
            icon=FluentIcon.BRUSH,
            text='切换主题',
            onClick=self.toggle_theme,
            selectable=False,
            position=NavigationItemPosition.BOTTOM
        )

        self.addSubInterface(
            self.settings_interface, 
            FluentIcon.SETTING, 
            "设置",
            NavigationItemPosition.BOTTOM
        )
        
    def connect_signals(self):
        # 语音设置应用
        self.voice_interface.apply_button.clicked.connect(self.apply_voice_settings)
        # 语音配置加载后自动应用
        self.voice_interface.config_loaded.connect(self.apply_voice_settings)
        
        # 文本编辑按钮
        self.text_interface.quick_run_button.clicked.connect(self.quick_run)
        self.text_interface.to_task_button.clicked.connect(self.to_task_plan)
        
        # 任务计划按钮
        self.task_interface.run_single_segment.connect(self.run_single_segment)
        self.task_interface.run_all_segments.connect(self.run_all_segments)
        self.task_interface.merge_audio.connect(self.merge_all_audio)
        self.task_interface.play_audio.connect(self.play_audio)
        
        # 监听配置变化
        self.task_interface.project_edit.textChanged.connect(
            lambda text: self.config_manager.set("project_name", text)
        )
        self.task_interface.output_edit.textChanged.connect(
            lambda text: self.config_manager.set("output_dir", text)
        )
    
    def on_theme_changed_in_nav(self, text):
        """侧边栏主题改变"""
        self.config_manager.set("theme", text)
        if text == "Light":
            setTheme(Theme.LIGHT)
        elif text == "Dark":
            setTheme(Theme.DARK)
        else:
            setTheme(Theme.AUTO)

    def load_initial_config(self):
        """加载初始配置"""
        # 加载项目名和输出目录
        project_name = self.config_manager.get("project_name", "project")
        output_dir = self.config_manager.get("output_dir", "./output")
        
        self.task_interface.project_edit.setText(project_name)
        self.task_interface.output_edit.setText(output_dir)
        self.task_interface.project_name = project_name
        self.task_interface.output_dir = output_dir
        
        # 自动加载上次的语音配置
        voice_config_path = self.config_manager.get("voice_config_path", "")
        if voice_config_path and os.path.exists(voice_config_path):
            self.voice_interface.load_config(voice_config_path)
        else:
            # 如果没有记录，尝试加载默认的 config/config.json
            default_config = "./config/config.json"
            if os.path.exists(default_config):
                self.voice_interface.load_config(default_config)
        
        # 确保初始配置被应用
        self.apply_voice_settings()

    def apply_voice_settings(self):
        """应用语音设置"""
        configs = self.voice_interface.get_voice_configs()
        self.text_interface.set_voice_configs(configs)
    
    def toggle_theme(self):
        """在Light和Dark之间切换主题"""
        from qfluentwidgets import qconfig
        if qconfig.theme == Theme.DARK:
            setTheme(Theme.LIGHT)
            self.config_manager.set("theme", "Light")
        else:
            setTheme(Theme.DARK)
            self.config_manager.set("theme", "Dark")
        
        InfoBar.success(
            title='成功',
            content='主题已切换',
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=1500,
            parent=self
        )
    
    def on_load_model_clicked(self):
        """手动加载模型"""
        if self.cosyvoice_model is not None:
            InfoBar.warning(
                title='模型已加载',
                content='CosyVoice 模型已经加载，无需重复加载。',
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            return
        
        # 创建并启动模型加载线程
        self.model_loader_thread = ModelLoaderThread()
        self.model_loader_thread.success.connect(self.on_model_loaded_success)
        self.model_loader_thread.error.connect(self.on_model_loaded_error)
        self.model_loader_thread.start()
    
    def on_model_loaded_success(self, model):
        """模型加载成功"""
        self.cosyvoice_model = model
        
        InfoBar.success(
            title='成功',
            content='CosyVoice 模型加载成功！',
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )
    
    def on_model_loaded_error(self, error_msg):
        """模型加载失败"""
        InfoBar.error(
            title='加载失败',
            content=f'模型加载失败: {error_msg[:50]}',
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self
        )
    
    def on_unload_model_clicked(self):
        """手动卸载模型"""
        if self.cosyvoice_model is None:
            InfoBar.warning(
                title='没有模型',
                content='当前没有加载任何模型。',
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            return
        
        # 检查是否有任务正在运行
        if self.current_worker and self.current_worker.isRunning():
            InfoBar.warning(
                title='任务正在运行',
                content='请等待当前任务完成后再卸载模型。',
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            return
        
        # 创建并启动模型卸载线程
        model_to_unload = self.cosyvoice_model
        self.cosyvoice_model = None  # 立即清空引用
        
        self.model_unloader_thread = ModelUnloaderThread(model_to_unload)
        self.model_unloader_thread.finished.connect(self.on_model_unloaded_success)
        self.model_unloader_thread.error.connect(self.on_model_unloaded_error)
        self.model_unloader_thread.start()
    
    def on_model_unloaded_success(self):
        """模型卸载成功"""
        InfoBar.success(
            title='成功',
            content='CosyVoice 模型已卸载！',
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )
    
    def on_model_unloaded_error(self, error_msg):
        """模型卸载失败"""
        InfoBar.error(
            title='卸载失败',
            content=f'模型卸载失败: {error_msg[:50]}',
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self
        )
    
    def quick_run(self):
        """一键运行"""
        segments = self.text_interface.get_text_segments()
        if not segments:
            InfoBar.warning(
                title="无内容",
                content="请输入文本并应用语音模式",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            return
        
        # 创建任务段落
        task_segments = [
            TaskSegment(i+1, text, config) 
            for i, (text, config) in enumerate(segments)
        ]
        
        # 开始生成
        self.start_generation(task_segments)
    
    def to_task_plan(self):
        """转到任务计划"""
        segments = self.text_interface.get_text_segments()
        if not segments:
            InfoBar.warning(
                title="无内容",
                content="请输入文本并应用语音模式",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            return
        
        # 加载到任务计划
        self.task_interface.load_segments(segments)
        
        # 切换到任务计划界面
        self.switchTo(self.task_interface)
        
        InfoBar.success(
            title="转换成功",
            content=f"已加载 {len(segments)} 个任务段落",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )
    
    def run_single_segment(self, index: int):
        """运行单个段落"""
        segment = self.task_interface.task_segments[index]
        self.task_interface.add_log(f"🚀 开始生成第 {segment.index} 段...")
        self.start_generation([segment])
    
    def run_all_segments(self):
        """运行所有段落"""
        segments = self.task_interface.task_segments
        if not segments:
            InfoBar.warning(
                title="无任务",
                content="请先添加任务段落",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            return
        
        self.task_interface.add_log(f"🚀 开始生成全部 {len(segments)} 段...")
        self.start_generation(segments)
    
    def start_generation(self, segments: List[TaskSegment]):
        """开始音频生成"""
        if self.current_worker and self.current_worker.isRunning():
            InfoBar.warning(
                title="正在运行",
                content="已有任务正在运行中",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            return
        
        # 创建工作线程
        self.current_worker = AudioGenerationWorker(
            segments,
            self.task_interface.output_dir,
            self.task_interface.project_name,
            self.cosyvoice_model
        )
        
        # 连接信号
        self.current_worker.progress.connect(self.task_interface.add_log)
        self.current_worker.segment_finished.connect(self.task_interface.update_segment_audio)
        self.current_worker.finished.connect(self.on_generation_finished)
        self.current_worker.error.connect(self.on_generation_error)
        
        # 禁用按钮
        self.task_interface.run_all_button.setEnabled(False)
        
        # 启动线程
        self.current_worker.start()
    
    def on_generation_finished(self, files: List[str]):
        """生成完成"""
        self.task_interface.add_log(f"🎉 生成完成！共 {len(files)} 个文件")
        
        # 更新模型引用
        if self.current_worker:
            self.cosyvoice_model = self.current_worker.cosyvoice
        
        # 恢复按钮
        self.task_interface.run_all_button.setEnabled(True)
        
        InfoBar.success(
            title="生成完成",
            content=f"成功生成 {len(files)} 个音频文件",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=5000,
            parent=self
        )
    
    def on_generation_error(self, error: str):
        """生成错误"""
        self.task_interface.add_log(f"❌ {error}")
        self.task_interface.run_all_button.setEnabled(True)
        
        InfoBar.error(
            title="生成失败",
            content=error,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=5000,
            parent=self
        )
    
    def merge_all_audio(self):
        """合成所有音频 - 按版本合成所有片段"""
        segments = self.task_interface.task_segments
        files_to_merge = []
        
        for segment in segments:
            if not segment.versions:
                continue
            
            # 获取当前选中的版本号
            version_idx = segment.current_version
            
            # 获取该版本的所有片段并按顺序添加
            if 0 <= version_idx < len(segment.versions):
                version_files = segment.versions[version_idx]
                files_to_merge.extend(version_files)
                
                # 日志输出
                if len(version_files) > 1:
                    self.task_interface.add_log(
                        f"📦 段落{segment.index}: v{version_idx+1} ({len(version_files)}个片段)"
                    )
                else:
                    self.task_interface.add_log(
                        f"📦 段落{segment.index}: v{version_idx+1}"
                    )
        
        if not files_to_merge:
            InfoBar.warning(
                title="无音频",
                content="没有可合成的音频文件",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
            return
        
        self.task_interface.add_log(f"🔧 开始合成 {len(files_to_merge)} 个音频片段...")
        
        # 合成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_file = merge_audio_files(
            files_to_merge, 
            self.task_interface.output_dir,
            f"{self.task_interface.project_name}_merged_{timestamp}.wav"
        )
        
        if merged_file:
            self.task_interface.add_log(f"✅ 合成完成: {os.path.basename(merged_file)}")
            InfoBar.success(
                title="合成完成",
                content=f"已保存到: {merged_file}",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=5000,
                parent=self
            )
        else:
            self.task_interface.add_log("❌ 合成失败")
            InfoBar.error(
                title="合成失败",
                content="音频合成时发生错误",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
    
    def play_audio(self, filepath: str):
        """播放音频"""
        if not os.path.exists(filepath):
            InfoBar.warning(
                title="文件不存在",
                content="音频文件不存在",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            return
        
        url = QUrl.fromLocalFile(filepath)
        self.media_player.setMedia(QMediaContent(url))
        self.media_player.play()
        
        self.task_interface.add_log(f"🔊 播放: {os.path.basename(filepath)}")
    
    def load_model_if_enabled(self):
        """如果设置中启用了自动加载，则加载模型"""
        auto_load = self.config_manager.get("auto_load_model", False)
        
        if not auto_load:
            return
        
        # 从 utils 模块加载函数
        from core.utils import load_cosyvoice_model
        
        try:
            self.cosyvoice_model = load_cosyvoice_model()
            # 显示成功提示
            InfoBar.success(
                title='模型加载成功',
                content="CosyVoice 模型已加载，现在可以生成语音了",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            InfoBar.warning(
                title='模型加载失败',
                content=f"未能加载 CosyVoice 模型，请检查模型文件",
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )