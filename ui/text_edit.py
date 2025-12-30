from typing import Dict, List, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QMenu, QApplication, QAction
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor

from qfluentwidgets import (
    PushButton, PrimaryPushButton, TextEdit, SubtitleLabel,
    BodyLabel, SimpleCardWidget, FluentIcon, InfoBar, InfoBarPosition,
    RoundMenu, Action
)

from core.models import VoiceConfig

class CustomTextEdit(TextEdit):
    """自定义文本编辑器"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.voice_configs: Dict[str, VoiceConfig] = {}
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 快捷指令定义（基于CosyVoice 2文档）
        self.quick_tags = {
            'strong': {'tag': '<strong>{}</strong>', 'name': '强调', 'shortcut': 'Alt+S'},
            'laughter': {'tag': '<laughter>{}</laughter>', 'name': '笑声', 'shortcut': 'Alt+L'},
            'breath': {'tag': '[breath]', 'name': '呼吸', 'shortcut': 'Alt+B'},
            'laugh_burst': {'tag': '[laughter]', 'name': '笑声爆发', 'shortcut': 'Alt+Shift+L'},
            'endofprompt': {'tag': '<|endofprompt|>', 'name': '指令结束符', 'shortcut': 'Alt+E'},
            'noise': {'tag': '[noise]', 'name': '噪音', 'shortcut': 'Alt+N'},
            'cough': {'tag': '[cough]', 'name': '咳嗽', 'shortcut': 'Alt+C'},
            'clucking': {'tag': '[clucking]', 'name': '咯咯声', 'shortcut': 'Alt+K'},
            'accent': {'tag': '[accent]', 'name': '口音', 'shortcut': 'Alt+A'},
            'quick_breath': {'tag': '[quick_breath]', 'name': '急促呼吸', 'shortcut': 'Alt+Q'},
            'hissing': {'tag': '[hissing]', 'name': '嘶嘶声', 'shortcut': 'Alt+H'},
            'sigh': {'tag': '[sigh]', 'name': '叹气', 'shortcut': 'Alt+I'},
            'vocalized_noise': {'tag': '[vocalized-noise]', 'name': '发声噪音', 'shortcut': 'Alt+V'},
            'lipsmack': {'tag': '[lipsmack]', 'name': '咂嘴', 'shortcut': 'Alt+P'},
            'mn': {'tag': '[mn]', 'name': '嗯', 'shortcut': 'Alt+M'},
            'endofsystem': {'tag': '<|endofsystem|>', 'name': '系统结束符', 'shortcut': 'Alt+Shift+E'},
        }
    
    def set_voice_configs(self, configs: Dict[str, VoiceConfig]):
        self.voice_configs = configs
    
    def keyPressEvent(self, event):
        # Ctrl+数字：应用语音配置
        if event.modifiers() == Qt.ControlModifier:
            key = event.key()
            if Qt.Key_1 <= key <= Qt.Key_9:
                mode_index = key - Qt.Key_1
                if self.textCursor().hasSelection():
                    config_list = list(self.voice_configs.values())
                    if mode_index < len(config_list):
                        config_name = config_list[mode_index].name
                        self.apply_voice_config(config_name)
                return
        
        # Alt+快捷键：插入控制标签
        if event.modifiers() == Qt.AltModifier:
            key = event.key()
            if key == Qt.Key_S:  # Alt+S: 强调
                self.insert_tag('strong')
                return
            elif key == Qt.Key_L:  # Alt+L: 笑声
                self.insert_tag('laughter')
                return
            elif key == Qt.Key_B:  # Alt+B: 呼吸
                self.insert_tag('breath')
                return
            elif key == Qt.Key_E:  # Alt+E: 指令结束符
                self.insert_tag('endofprompt')
                return
            elif key == Qt.Key_N:  # Alt+N: 噪音
                self.insert_tag('noise')
                return
            elif key == Qt.Key_C:  # Alt+C: 咳嗽
                self.insert_tag('cough')
                return
            elif key == Qt.Key_K:  # Alt+K: 咯咯声
                self.insert_tag('clucking')
                return
            elif key == Qt.Key_A:  # Alt+A: 口音
                self.insert_tag('accent')
                return
            elif key == Qt.Key_Q:  # Alt+Q: 急促呼吸
                self.insert_tag('quick_breath')
                return
            elif key == Qt.Key_H:  # Alt+H: 嘶嘶声
                self.insert_tag('hissing')
                return
            elif key == Qt.Key_I:  # Alt+I: 叹气
                self.insert_tag('sigh')
                return
            elif key == Qt.Key_V:  # Alt+V: 发声噪音
                self.insert_tag('vocalized_noise')
                return
            elif key == Qt.Key_P:  # Alt+P: 咂嘴
                self.insert_tag('lipsmack')
                return
            elif key == Qt.Key_M:  # Alt+M: 嗯
                self.insert_tag('mn')
                return
        
        # Alt+Shift+L: 笑声爆发
        if event.modifiers() == (Qt.AltModifier | Qt.ShiftModifier):
            if event.key() == Qt.Key_L:
                self.insert_tag('laugh_burst')
                return
            elif event.key() == Qt.Key_E:
                self.insert_tag('endofsystem')
                return
        
        super().keyPressEvent(event)
    
    def show_context_menu(self, position: QPoint):
        menu = RoundMenu(parent=self)
        
        if self.textCursor().hasSelection():
            menu.addAction(Action(FluentIcon.COPY, "复制", self, triggered=self.copy))
            menu.addAction(Action(FluentIcon.CUT, "剪切", self, triggered=self.cut))
        
        menu.addAction(Action(FluentIcon.PASTE, "粘贴", self, triggered=self.paste))
        
        menu.addSeparator()
        
        menu.addAction(Action(FluentIcon.TILES, "全选", self, triggered=self.selectAll))
        
        # 快捷指令菜单
        if self.textCursor().hasSelection() or True:
            menu.addSeparator()
            tag_menu = RoundMenu("🏷️ 插入控制标签", self)
            tag_menu.setIcon(FluentIcon.TAG)
            menu.addMenu(tag_menu)
            
            for tag_key, tag_info in self.quick_tags.items():
                action = Action(FluentIcon.TAG, f"{tag_info['name']} ({tag_info['shortcut']})", self)
                action.triggered.connect(lambda checked, key=tag_key: self.insert_tag(key))
                tag_menu.addAction(action)
        
        # 语音配置菜单
        if self.textCursor().hasSelection() and self.voice_configs:
            menu.addSeparator()
            voice_menu = RoundMenu("🎤 应用语音配置", self)
            voice_menu.setIcon(FluentIcon.MICROPHONE)
            menu.addMenu(voice_menu)
            
            for i, (config_name, config) in enumerate(self.voice_configs.items()):
                action = Action(FluentIcon.PEOPLE, f"Ctrl+{i+1}: {config_name} ({config.mode})", self)
                action.triggered.connect(lambda checked, name=config_name: self.apply_voice_config(name))
                voice_menu.addAction(action)
        
        menu.exec_(self.mapToGlobal(position))
    
    def insert_tag(self, tag_key: str):
        """插入控制标签"""
        if tag_key not in self.quick_tags:
            return
        
        tag_info = self.quick_tags[tag_key]
        cursor = self.textCursor()
        
        if cursor.hasSelection():
            # 选中文本时，用标签包裹
            selected_text = cursor.selectedText()
            if '{}' in tag_info['tag']:
                new_text = tag_info['tag'].format(selected_text)
            else:
                new_text = tag_info['tag'] + selected_text
            cursor.insertText(new_text)
        else:
            # 未选中时，直接插入标签
            if '{}' in tag_info['tag']:
                # 有占位符的标签，插入后选中中间部分
                tag_parts = tag_info['tag'].split('{}')
                cursor.insertText(tag_parts[0])
                start_pos = cursor.position()
                cursor.insertText(tag_parts[1])
                cursor.setPosition(start_pos)
            else:
                cursor.insertText(tag_info['tag'])
    
    def apply_voice_config(self, config_name: str):
        if config_name not in self.voice_configs:
            return
        
        config = self.voice_configs[config_name]
        cursor = self.textCursor()
        
        if cursor.hasSelection():
            char_format = QTextCharFormat()
            char_format.setBackground(QColor(config.color))
            char_format.setProperty(QTextCharFormat.UserProperty, config_name)
            cursor.mergeCharFormat(char_format)
    
    def get_text_segments(self) -> List[Tuple[str, VoiceConfig]]:
        """提取按颜色分段的文本段落"""
        segments = []
        document = self.document()
        full_text = self.toPlainText()
        
        if not full_text.strip():
            return segments
        
        current_segment = ""
        current_config = None
        
        for i in range(len(full_text)):
            cursor = QTextCursor(document)
            cursor.setPosition(i)
            cursor.setPosition(i + 1, QTextCursor.KeepAnchor)
            
            char = full_text[i]
            char_format = cursor.charFormat()
            config_name = char_format.property(QTextCharFormat.UserProperty)
            
            if config_name and config_name in self.voice_configs:
                char_config = self.voice_configs[config_name]
            else:
                if self.voice_configs:
                    char_config = list(self.voice_configs.values())[0]
                else:
                    continue
            
            if current_config is not None and (
                current_config.name != char_config.name or 
                char == '\n'
            ):
                if current_segment.strip():
                    segments.append((current_segment.strip(), current_config))
                current_segment = ""
                current_config = None
            
            if char == '\n':
                continue
                
            if char.strip():
                if current_config is None:
                    current_config = char_config
                current_segment += char
            elif current_segment:
                current_segment += char
        
        if current_segment.strip() and current_config:
            segments.append((current_segment.strip(), current_config))
        
        return segments


class TextEditInterface(QWidget):
    """文本编辑界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        title = SubtitleLabel("📝 文本编辑")
        layout.addWidget(title)
        
        # 文本编辑区
        text_card = SimpleCardWidget()
        text_layout = QVBoxLayout(text_card)
        
        text_label = BodyLabel("输入文本内容 (右键或Ctrl+数字应用语音模式):")
        text_layout.addWidget(text_label)
        
        self.text_edit = CustomTextEdit()
        self.text_edit.setPlaceholderText(
            "请输入要转换为语音的文本内容...\n\n"
            "使用右键菜单或Ctrl+数字快捷键为选中的文本应用不同的语音模式。\n"
            "不同颜色代表不同的语音配置。"
        )
        text_layout.addWidget(self.text_edit)
        
        layout.addWidget(text_card)
        
        # 按钮区
        button_layout = QHBoxLayout()
        
        self.quick_run_button = PrimaryPushButton("⚡ 一键运行")
        self.quick_run_button.setToolTip("直接按顺序推理生成所有音频")
        button_layout.addWidget(self.quick_run_button)
        
        self.to_task_button = PushButton("📋 转成计划任务")
        self.to_task_button.setToolTip("转到任务计划界面，可以单独调整每段的参数")
        button_layout.addWidget(self.to_task_button)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def set_voice_configs(self, configs: Dict[str, VoiceConfig]):
        self.text_edit.set_voice_configs(configs)
    
    def get_text_segments(self) -> List[Tuple[str, VoiceConfig]]:
        return self.text_edit.get_text_segments()
