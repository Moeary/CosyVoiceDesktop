from typing import Any, Dict, List, Tuple

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QHeaderView, QTableWidgetItem, QTextEdit as QtTextEdit, QStackedWidget, QMessageBox
from PyQt5.QtCore import Qt, QPoint, QEvent, pyqtSignal
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor

from qfluentwidgets import PushButton, PrimaryPushButton, TextEdit, SubtitleLabel, BodyLabel, SimpleCardWidget, FluentIcon, RoundMenu, Action, TableWidget, ComboBox, InfoBar, InfoBarPosition

from core.models import VoiceConfig


UNMAPPED_VOICE_OPTION = "未映射"


class CustomTextEdit(TextEdit):
    """自定义文本编辑器"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.voice_configs: Dict[str, VoiceConfig] = {}
        self.default_config_name = ""
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

    def set_default_voice_config(self, config_name: str):
        self.default_config_name = (config_name or "").strip()

    def get_fallback_config(self):
        if self.default_config_name and self.default_config_name in self.voice_configs:
            return self.voice_configs[self.default_config_name]
        if self.voice_configs:
            return list(self.voice_configs.values())[0]
        return None

    def get_fallback_config_name(self) -> str:
        fallback = self.get_fallback_config()
        return fallback.name if fallback else ""
    
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

    def insertFromMimeData(self, source):
        if source and source.hasText():
            text = source.text().replace('\u2028', '\n').replace('\u2029', '\n')
            self.textCursor().insertText(text)
            return
        super().insertFromMimeData(source)
    
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
            tag_menu = RoundMenu("插入控制标签", self)
            tag_menu.setIcon(FluentIcon.TAG)
            menu.addMenu(tag_menu)
            
            for tag_key, tag_info in self.quick_tags.items():
                action = Action(FluentIcon.TAG, f"{tag_info['name']} ({tag_info['shortcut']})", self)
                action.triggered.connect(lambda checked, key=tag_key: self.insert_tag(key))
                tag_menu.addAction(action)
        
        # 语音配置菜单
        if self.textCursor().hasSelection() and self.voice_configs:
            menu.addSeparator()
            voice_menu = RoundMenu("应用语音配置", self)
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
        
        cursor = self.textCursor()
        
        if cursor.hasSelection():
            self.apply_voice_config_to_range(
                config_name,
                cursor.selectionStart(),
                cursor.selectionEnd()
            )

    def apply_voice_config_to_range(self, config_name: str, start: int, end: int):
        if config_name not in self.voice_configs or start >= end:
            return

        config = self.voice_configs[config_name]
        cursor = self.textCursor()
        cursor.beginEditBlock()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.KeepAnchor)

        char_format = QTextCharFormat()
        char_format.setBackground(QColor(config.color))
        char_format.setProperty(QTextCharFormat.UserProperty, config_name)
        cursor.mergeCharFormat(char_format)
        cursor.endEditBlock()

    def clear_voice_labels(self, start: int = None, end: int = None):
        full_text = self.toPlainText()
        if not full_text:
            return

        start = 0 if start is None else start
        end = len(full_text) if end is None else end
        if start >= end:
            return

        cursor = self.textCursor()
        cursor.beginEditBlock()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.KeepAnchor)

        clear_format = QTextCharFormat()
        clear_format.setBackground(QColor(Qt.transparent))
        clear_format.setProperty(QTextCharFormat.UserProperty, "")
        cursor.mergeCharFormat(clear_format)
        cursor.endEditBlock()

    def get_voice_config_name_at_position(self, position: int) -> str:
        full_text = self.toPlainText()
        if position < 0 or position >= len(full_text):
            return ""

        cursor = QTextCursor(self.document())
        cursor.setPosition(position)
        cursor.setPosition(position + 1, QTextCursor.KeepAnchor)
        config_name = cursor.charFormat().property(QTextCharFormat.UserProperty)

        if config_name and config_name in self.voice_configs:
            return config_name
        return ""

    def get_block_voice_config_name(self, start: int, end: int) -> str:
        full_text = self.toPlainText()
        for position in range(start, end):
            if full_text[position].strip():
                return self.get_voice_config_name_at_position(position)
        return ""

    def get_assignable_blocks(self) -> List[dict]:
        blocks = []
        document = self.document()
        block = document.begin()

        while block.isValid():
            raw_text = block.text()
            trimmed_text = raw_text.strip()
            if trimmed_text:
                start = block.position()
                end = start + len(raw_text)
                blocks.append({
                    'index': len(blocks) + 1,
                    'block_number': block.blockNumber(),
                    'start': start,
                    'end': end,
                    'text': trimmed_text,
                    'current_speaker': self.get_block_voice_config_name(start, end)
                })
            block = block.next()

        return blocks

    def build_manual_assignments(self) -> List[dict]:
        assignments = []
        fallback_name = self.get_fallback_config_name()
        for block in self.get_assignable_blocks():
            speaker = block['current_speaker'] or fallback_name
            if not speaker:
                continue
            assignments.append({
                'index': block['index'],
                'speaker': speaker,
                'text': block['text'],
                'category': 'manual' if block['current_speaker'] else 'default_fallback',
                'reason': '当前文本已标注该角色' if block['current_speaker'] else '当前文本未显式标注，使用默认角色',
                'confidence': None,
                'current_speaker': block['current_speaker'],
                'start': block['start'],
                'end': block['end'],
            })
        return assignments

    def apply_block_assignments(self, assignments: List[dict], clear_existing: bool = True):
        if clear_existing:
            self.clear_voice_labels()

        blocks = self.get_assignable_blocks()
        block_map = {block['index']: block for block in blocks}

        for assignment in assignments:
            speaker = assignment.get('speaker', '')
            start = assignment.get('start')
            end = assignment.get('end')

            if speaker not in self.voice_configs:
                continue

            if isinstance(start, int) and isinstance(end, int) and start < end:
                self.apply_voice_config_to_range(speaker, start, end)
                continue

            try:
                index = int(assignment.get('index'))
            except (TypeError, ValueError):
                continue

            block = block_map.get(index)
            if block:
                self.apply_voice_config_to_range(speaker, block['start'], block['end'])

    def highlight_block_indices(self, indices: List[Any]):
        block_map = {block['index']: block for block in self.get_assignable_blocks()}
        selections = []
        for target in indices:
            if isinstance(target, dict):
                start = target.get('start')
                end = target.get('end')
                if not (isinstance(start, int) and isinstance(end, int) and start < end):
                    block = block_map.get(target.get('index'))
                    if not block:
                        continue
                    start, end = block['start'], block['end']
            else:
                block = block_map.get(target)
                if not block:
                    continue
                start, end = block['start'], block['end']

            selection = QtTextEdit.ExtraSelection()
            cursor = QTextCursor(self.document())
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.KeepAnchor)
            selection.cursor = cursor
            selection.format.setBackground(QColor(255, 209, 102, 120))
            selection.format.setUnderlineStyle(QTextCharFormat.SingleUnderline)
            selection.format.setUnderlineColor(QColor(255, 140, 0))
            selections.append(selection)
        self.setExtraSelections(selections)

    def clear_highlight_preview(self):
        self.setExtraSelections([])

    def _capture_text_and_voice_labels(self) -> Tuple[str, List[str]]:
        full_text = self.toPlainText()
        labels: List[str] = []
        document = self.document()

        for index in range(len(full_text)):
            cursor = QTextCursor(document)
            cursor.setPosition(index)
            cursor.setPosition(index + 1, QTextCursor.KeepAnchor)
            config_name = cursor.charFormat().property(QTextCharFormat.UserProperty)
            labels.append(config_name if config_name in self.voice_configs else "")

        return full_text, labels

    def _replace_text_with_labels(self, text: str, labels: List[str]):
        if len(text) != len(labels):
            raise ValueError("文本长度与标签长度不一致")

        old_cursor = self.textCursor()
        old_position = min(old_cursor.position(), len(text))

        self.setPlainText(text)
        self.clear_highlight_preview()
        reset_cursor = QTextCursor(self.document())
        reset_cursor.select(QTextCursor.Document)
        reset_cursor.setCharFormat(QTextCharFormat())

        if text:
            run_start = 0
            current_label = labels[0]
            for index in range(1, len(text) + 1):
                next_label = labels[index] if index < len(text) else None
                if next_label != current_label:
                    if current_label in self.voice_configs and run_start < index:
                        self.apply_voice_config_to_range(current_label, run_start, index)
                    if index < len(text):
                        run_start = index
                        current_label = next_label

        cursor = self.textCursor()
        cursor.setPosition(old_position)
        self.setTextCursor(cursor)

    def normalize_text_content(self) -> bool:
        full_text, labels = self._capture_text_and_voice_labels()
        if not full_text:
            return False

        new_chars: List[str] = []
        new_labels: List[str] = []
        pending_space = False

        for char, label in zip(full_text, labels):
            if char in '\r\n':
                if new_chars and not new_chars[-1].isspace():
                    pending_space = True
                continue

            normalized_char = ' ' if char == '\t' else char
            if pending_space:
                if normalized_char.isspace():
                    continue
                new_chars.append(' ')
                new_labels.append(new_labels[-1] if new_labels else '')
                pending_space = False

            new_chars.append(normalized_char)
            new_labels.append(label)

        start = 0
        end = len(new_chars)
        while start < end and new_chars[start].isspace():
            start += 1
        while end > start and new_chars[end - 1].isspace():
            end -= 1

        self._replace_text_with_labels(''.join(new_chars[start:end]), new_labels[start:end])
        return True

    def _get_markup_tokens(self) -> List[str]:
        tokens = set()
        for tag_info in self.quick_tags.values():
            tag = tag_info['tag']
            if '{}' in tag:
                open_tag, close_tag = tag.split('{}')
                if open_tag:
                    tokens.add(open_tag)
                if close_tag:
                    tokens.add(close_tag)
            else:
                tokens.add(tag)
        return sorted(tokens, key=len, reverse=True)

    def strip_markup_tokens(self) -> bool:
        full_text, labels = self._capture_text_and_voice_labels()
        if not full_text:
            return False

        tokens = self._get_markup_tokens()
        new_chars: List[str] = []
        new_labels: List[str] = []
        index = 0

        while index < len(full_text):
            matched_token = next((token for token in tokens if full_text.startswith(token, index)), None)
            if matched_token:
                index += len(matched_token)
                continue

            new_chars.append(full_text[index])
            new_labels.append(labels[index])
            index += 1

        self._replace_text_with_labels(''.join(new_chars), new_labels)
        return True
    
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
                char_config = self.get_fallback_config()
                if char_config is None:
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


class RoleConsolePanel(SimpleCardWidget):
    """文本编辑页中的角色控制台分组面板"""

    applyAssignmentsRequested = pyqtSignal(list, bool)
    highlightIndicesRequested = pyqtSignal(list)
    clearHighlightRequested = pyqtSignal()

    def __init__(self, title: str, description: str, empty_message: str, apply_all_text: str, parent=None):
        super().__init__(parent)
        self.voice_config_names: List[str] = []
        self.group_entries: List[dict] = []
        self.current_group_index = -1
        self.empty_message = empty_message
        self.apply_all_clears_existing = True

        layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()
        self.title_label = SubtitleLabel(title)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        self.description_label = BodyLabel(description)
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: gray;")
        layout.addWidget(self.description_label)

        self.extra_button_layout = QHBoxLayout()
        self.extra_button_layout.addStretch()
        layout.addLayout(self.extra_button_layout)

        self.group_table = TableWidget(self)
        self.group_table.setColumnCount(3)
        self.group_table.setHorizontalHeaderLabels(["识别分组", "段落数", "首句预览"])
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.setMouseTracking(True)
        self.group_table.viewport().installEventFilter(self)
        group_header = self.group_table.horizontalHeader()
        group_header.setSectionResizeMode(QHeaderView.Interactive)
        group_header.setStretchLastSection(True)
        group_header.setMinimumSectionSize(70)
        self.group_table.setColumnWidth(0, 180)
        self.group_table.setColumnWidth(1, 70)
        self.group_table.currentCellChanged.connect(self.on_group_changed)
        self.group_table.cellEntered.connect(self.on_group_hovered)
        layout.addWidget(self.group_table, 1)

        mapping_layout = QHBoxLayout()
        self.mapping_label = BodyLabel("应用到角色:")
        self.mapping_combo = ComboBox(self)
        self.mapping_combo.currentTextChanged.connect(self.on_mapping_changed)
        mapping_layout.addWidget(self.mapping_label)
        mapping_layout.addWidget(self.mapping_combo, 1)
        layout.addLayout(mapping_layout)

        action_layout = QHBoxLayout()
        self.apply_selected_button = PushButton("应用选中角色")
        self.apply_selected_button.clicked.connect(self.apply_selected_group)
        action_layout.addWidget(self.apply_selected_button)

        self.apply_all_button = PrimaryPushButton(apply_all_text)
        self.apply_all_button.clicked.connect(self.apply_all_groups)
        action_layout.addWidget(self.apply_all_button)
        layout.addLayout(action_layout)

        self.segment_table = TableWidget(self)
        self.segment_table.setColumnCount(4)
        self.segment_table.setHorizontalHeaderLabels(["段落", "文本内容", "属性", "说明"])
        self.segment_table.verticalHeader().setVisible(False)
        self.segment_table.setMouseTracking(True)
        self.segment_table.viewport().installEventFilter(self)
        segment_header = self.segment_table.horizontalHeader()
        segment_header.setSectionResizeMode(QHeaderView.Interactive)
        segment_header.setStretchLastSection(True)
        segment_header.setMinimumSectionSize(70)
        self.segment_table.setColumnWidth(0, 60)
        self.segment_table.setColumnWidth(1, 250)
        self.segment_table.setColumnWidth(2, 90)
        self.segment_table.currentCellChanged.connect(self.on_segment_changed)
        self.segment_table.cellEntered.connect(self.on_segment_hovered)
        layout.addWidget(self.segment_table, 2)

        self._update_state()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Leave:
            watched_viewports = set()
            group_table = getattr(self, 'group_table', None)
            segment_table = getattr(self, 'segment_table', None)
            if group_table is not None:
                watched_viewports.add(group_table.viewport())
            if segment_table is not None:
                watched_viewports.add(segment_table.viewport())
            if obj in watched_viewports:
                self.clearHighlightRequested.emit()
        return super().eventFilter(obj, event)

    def add_header_button(self, button):
        self.extra_button_layout.insertWidget(self.extra_button_layout.count() - 1, button)

    def set_voice_config_names(self, names: List[str]):
        self.voice_config_names = list(names)
        self.refresh_mapping_combo()

    def set_assignments(self, assignments: List[dict]):
        old_mapping = {entry['group_name']: entry['mapped_speaker'] for entry in self.group_entries}
        grouped: Dict[str, List[dict]] = {}
        for item in assignments:
            group_name = str(
                item.get('group_name')
                or item.get('speaker_label')
                or item.get('speaker')
                or '未分配'
            ).strip()
            grouped.setdefault(group_name, []).append(item)

        self.group_entries = []
        for group_name, grouped_items in grouped.items():
            items = sorted(grouped_items, key=lambda item: item.get('index', 0))
            suggested_voice = str(items[0].get('suggested_voice', '') or '').strip() if items else ""
            if suggested_voice not in self.voice_config_names:
                suggested_voice = ""
            mapped_speaker = old_mapping.get(group_name) or suggested_voice
            if mapped_speaker not in self.voice_config_names:
                mapped_speaker = group_name if group_name in self.voice_config_names else ""
            self.group_entries.append({
                'group_name': group_name,
                'mapped_speaker': mapped_speaker,
                'suggested_voice': suggested_voice,
                'items': items,
            })

        self.update_group_table()
        self._update_state()

    def clear_assignments(self):
        self.group_entries = []
        self.group_table.setRowCount(0)
        self.segment_table.setRowCount(0)
        self.current_group_index = -1
        self._update_state()

    def refresh_mapping_combo(self):
        current_text = self.mapping_combo.currentText()
        self.mapping_combo.blockSignals(True)
        self.mapping_combo.clear()
        self.mapping_combo.addItems([UNMAPPED_VOICE_OPTION] + self.voice_config_names)
        if current_text in self.voice_config_names:
            self.mapping_combo.setCurrentText(current_text)
        else:
            self.mapping_combo.setCurrentText(UNMAPPED_VOICE_OPTION)
        self.mapping_combo.blockSignals(False)
        self.sync_mapping_from_selection()

    def _update_state(self):
        has_data = bool(self.group_entries)
        self.mapping_label.setEnabled(has_data)
        self.mapping_combo.setEnabled(has_data and bool(self.voice_config_names))
        self.apply_selected_button.setEnabled(bool(self.build_assignments_for_group(self.current_group_index)))
        self.apply_all_button.setEnabled(bool(self.build_assignments_for_all_groups()))
        if not has_data:
            self.segment_table.setRowCount(0)

    def update_group_table(self):
        self.group_table.setRowCount(len(self.group_entries))
        for row, entry in enumerate(self.group_entries):
            preview = entry['items'][0].get('text', '') if entry['items'] else self.empty_message
            preview = preview.replace('\n', ' ')
            self.group_table.setItem(row, 0, QTableWidgetItem(entry['group_name']))
            self.group_table.setItem(row, 1, QTableWidgetItem(str(len(entry['items']))))
            preview_item = QTableWidgetItem(preview)
            preview_item.setToolTip(preview)
            self.group_table.setItem(row, 2, preview_item)

        if self.group_entries:
            self.group_table.selectRow(0)
            self.on_group_changed(0, 0, -1, -1)
        else:
            self.current_group_index = -1
            self.segment_table.setRowCount(0)

    def on_group_changed(self, current_row, current_column, previous_row, previous_column):
        del current_column, previous_row, previous_column
        self.current_group_index = current_row if 0 <= current_row < len(self.group_entries) else -1
        self.update_segment_table()
        self.sync_mapping_from_selection()
        self._update_state()

    def on_group_hovered(self, row, column):
        del column
        if 0 <= row < len(self.group_entries):
            self.highlightIndicesRequested.emit(self.group_entries[row]['items'])

    def update_segment_table(self):
        if self.current_group_index < 0 or self.current_group_index >= len(self.group_entries):
            self.segment_table.setRowCount(0)
            return

        items = self.group_entries[self.current_group_index]['items']
        self.segment_table.setRowCount(len(items))
        for row, item in enumerate(items):
            category = item.get('category', '') or '-'
            reason = item.get('reason', '') or '-'
            speaker_label = item.get('speaker_label') or item.get('raw_speaker') or ''
            if speaker_label and category in {'dialogue', 'thought'} and speaker_label not in reason:
                prefix = "说话人" if category == 'dialogue' else "思考人"
                reason = f"{prefix}: {speaker_label}" if reason == '-' else f"{prefix}: {speaker_label} | {reason}"
            confidence = item.get('confidence')
            if confidence is not None:
                reason = f"{reason} | 置信度 {confidence:.2f}"

            self.segment_table.setItem(row, 0, QTableWidgetItem(str(item.get('index', row + 1))))
            text_item = QTableWidgetItem(item.get('text', '').replace('\n', ' '))
            text_item.setToolTip(item.get('text', ''))
            self.segment_table.setItem(row, 1, text_item)
            self.segment_table.setItem(row, 2, QTableWidgetItem(category))
            reason_item = QTableWidgetItem(reason)
            reason_item.setToolTip(reason)
            self.segment_table.setItem(row, 3, reason_item)

    def on_segment_changed(self, current_row, current_column, previous_row, previous_column):
        del current_column, previous_row, previous_column
        self.highlight_single_segment(current_row)

    def on_segment_hovered(self, row, column):
        del column
        self.highlight_single_segment(row)

    def highlight_single_segment(self, row: int):
        if self.current_group_index < 0 or self.current_group_index >= len(self.group_entries):
            return
        items = self.group_entries[self.current_group_index]['items']
        if 0 <= row < len(items):
            self.highlightIndicesRequested.emit([items[row]])

    def sync_mapping_from_selection(self):
        self.mapping_combo.blockSignals(True)
        self.mapping_combo.clear()
        self.mapping_combo.addItems([UNMAPPED_VOICE_OPTION] + self.voice_config_names)
        if self.current_group_index >= 0 and self.current_group_index < len(self.group_entries):
            mapped = self.group_entries[self.current_group_index]['mapped_speaker']
            if mapped in self.voice_config_names:
                self.mapping_combo.setCurrentText(mapped)
            else:
                self.mapping_combo.setCurrentText(UNMAPPED_VOICE_OPTION)
        else:
            self.mapping_combo.setCurrentText(UNMAPPED_VOICE_OPTION)
        self.mapping_combo.blockSignals(False)

    def on_mapping_changed(self, text: str):
        if self.current_group_index < 0 or self.current_group_index >= len(self.group_entries):
            return
        self.group_entries[self.current_group_index]['mapped_speaker'] = "" if text == UNMAPPED_VOICE_OPTION else text
        self._update_state()

    def build_assignments_for_group(self, group_index: int) -> List[dict]:
        if group_index < 0 or group_index >= len(self.group_entries):
            return []
        entry = self.group_entries[group_index]
        speaker = entry['mapped_speaker']
        if not speaker:
            return []
        assignments = []
        for item in entry['items']:
            assignment = {'speaker': speaker}
            start = item.get('start')
            end = item.get('end')
            if isinstance(start, int) and isinstance(end, int) and start < end:
                assignment.update({'start': start, 'end': end})
            else:
                assignment['index'] = item.get('index')
            assignments.append(assignment)
        return assignments

    def build_assignments_for_all_groups(self) -> List[dict]:
        assignments = []
        for group_index in range(len(self.group_entries)):
            assignments.extend(self.build_assignments_for_group(group_index))
        return sorted(assignments, key=lambda item: item.get('index', 0))

    def apply_selected_group(self):
        assignments = self.build_assignments_for_group(self.current_group_index)
        if assignments:
            self.applyAssignmentsRequested.emit(assignments, False)

    def apply_all_groups(self):
        assignments = self.build_assignments_for_all_groups()
        if assignments:
            self.applyAssignmentsRequested.emit(assignments, self.apply_all_clears_existing)


class TextEditInterface(QWidget):
    """文本编辑界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_console_mode = "manual"
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(12)
        
        title = SubtitleLabel("文本编辑")
        layout.addWidget(title)

        main_splitter = QSplitter(Qt.Horizontal, self)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(8)

        text_card = SimpleCardWidget()
        text_card.setMinimumWidth(560)
        text_layout = QVBoxLayout(text_card)
        text_layout.setContentsMargins(18, 18, 18, 18)
        text_layout.setSpacing(12)

        text_label = BodyLabel("输入文本内容（右键 / Ctrl+数字手动标注；右侧控制台可查看并应用手动或 AI 角色结果）:")
        text_label.setWordWrap(True)
        text_layout.addWidget(text_label)

        self.text_edit = CustomTextEdit()
        self.text_edit.setMinimumHeight(380)
        self.text_edit.setPlaceholderText(
            "请输入要转换为语音的文本内容...\n\n"
            "使用右键菜单或 Ctrl+数字 快捷键为选中的文本应用不同的语音模式。\n"
            "右侧角色控制台可以读取当前手动标签，或显示 AI 自动识别出的角色分组。\n"
            "鼠标悬停在控制台的角色/句子上时，会在文本编辑框中高亮对应内容。"
        )
        text_layout.addWidget(self.text_edit)

        button_layout = QHBoxLayout()
        self.quick_run_button = PrimaryPushButton("一键运行")
        self.quick_run_button.setToolTip("直接按顺序推理生成所有音频")
        button_layout.addWidget(self.quick_run_button)

        self.to_task_button = PushButton("转成计划任务")
        self.to_task_button.setToolTip("转到任务计划界面，可以单独调整每段的参数")
        button_layout.addWidget(self.to_task_button)

        self.normalize_button = PushButton("规范化文本")
        self.normalize_button.setToolTip("移除换行和字体格式，保留角色打标")
        self.normalize_button.clicked.connect(self.normalize_text_content)
        button_layout.addWidget(self.normalize_button)

        self.format_button = PushButton("格式化文本")
        self.format_button.setToolTip("去掉控制标签内容前会弹出确认")
        self.format_button.clicked.connect(self.format_text_content)
        button_layout.addWidget(self.format_button)

        button_layout.addStretch()
        text_layout.addLayout(button_layout)

        console_card = SimpleCardWidget()
        console_card.setMinimumWidth(600)
        console_layout = QVBoxLayout(console_card)
        console_layout.setContentsMargins(18, 18, 18, 18)
        console_layout.setSpacing(12)

        console_header = QHBoxLayout()
        console_title = SubtitleLabel("角色控制台")
        console_header.addWidget(console_title)
        console_header.addStretch()
        self.console_toggle_button = PushButton("切换到AI分组")
        self.console_toggle_button.clicked.connect(self.toggle_console_panel)
        console_header.addWidget(self.console_toggle_button)
        self.ai_assign_button = PrimaryPushButton("AI分配角色")
        self.ai_assign_button.setToolTip("调用 OpenAI 兼容大模型按全文自动断句并识别角色")
        console_header.addWidget(self.ai_assign_button)
        console_layout.addLayout(console_header)

        console_tip = BodyLabel("可在手动分组和 AI分组之间切换查看，按角色预览句子、高亮文本并应用映射。")
        console_tip.setWordWrap(True)
        console_tip.setStyleSheet("color: gray;")
        console_layout.addWidget(console_tip)

        self.console_stack = QStackedWidget(console_card)

        self.manual_panel = RoleConsolePanel(
            "手动分组",
            "把当前文本里已有的角色标签同步到控制台。可按角色预览、高亮、替换映射并写回文本。",
            "当前还没有手动分组",
            "手动结果写回文本"
        )
        self.manual_read_button = PushButton("从文本读取")
        self.manual_read_button.clicked.connect(lambda: self.load_manual_assignments_from_text(show_panel=True))
        self.manual_panel.add_header_button(self.manual_read_button)

        self.ai_panel = RoleConsolePanel(
            "AI分组",
            "展示 AI 抽取出的 category / speaker 分组，例如 dialogue / 悟空、thought / 悟空、narration，再决定映射到哪个本地配音配置。",
            "当前还没有 AI 分组",
            "应用全部 AI 结果"
        )

        self.console_stack.addWidget(self.manual_panel)
        self.console_stack.addWidget(self.ai_panel)
        console_layout.addWidget(self.console_stack)

        main_splitter.addWidget(text_card)
        main_splitter.addWidget(console_card)
        main_splitter.setStretchFactor(0, 4)
        main_splitter.setStretchFactor(1, 5)
        main_splitter.setSizes([760, 700])
        layout.addWidget(main_splitter, 1)

        self.manual_panel.applyAssignmentsRequested.connect(self.apply_voice_assignments)
        self.ai_panel.applyAssignmentsRequested.connect(self.apply_voice_assignments)
        self.manual_panel.highlightIndicesRequested.connect(self.text_edit.highlight_block_indices)
        self.ai_panel.highlightIndicesRequested.connect(self.text_edit.highlight_block_indices)
        self.manual_panel.clearHighlightRequested.connect(self.text_edit.clear_highlight_preview)
        self.ai_panel.clearHighlightRequested.connect(self.text_edit.clear_highlight_preview)
        self.show_ai_panel()
    
    def set_voice_configs(self, configs: Dict[str, VoiceConfig]):
        self.text_edit.set_voice_configs(configs)
        names = list(configs.keys())
        self.manual_panel.set_voice_config_names(names)
        self.ai_panel.set_voice_config_names(names)

    def set_default_voice_config(self, config_name: str):
        self.text_edit.set_default_voice_config(config_name)

    def show_manual_panel(self):
        self.current_console_mode = "manual"
        self.console_stack.setCurrentWidget(self.manual_panel)
        self.console_toggle_button.setText("切换到AI分组")

    def show_ai_panel(self):
        self.current_console_mode = "ai"
        self.console_stack.setCurrentWidget(self.ai_panel)
        self.console_toggle_button.setText("切换到手动分组")

    def toggle_console_panel(self):
        if self.current_console_mode == "manual":
            self.show_ai_panel()
        else:
            self.show_manual_panel()
    
    def get_text_segments(self) -> List[Tuple[str, VoiceConfig]]:
        return self.text_edit.get_text_segments()

    def get_assignable_blocks(self) -> List[dict]:
        return self.text_edit.get_assignable_blocks()

    def get_plain_text(self) -> str:
        return self.text_edit.toPlainText()

    def load_manual_assignments_from_text(self, show_panel: bool = False):
        self.refresh_manual_assignments(show_panel=show_panel)

    def set_ai_assignments(self, assignments: List[dict]):
        self.ai_panel.set_assignments(assignments)
        self.show_ai_panel()

    def clear_ai_assignments(self, show_manual: bool = False):
        self.ai_panel.clear_assignments()
        if show_manual and self.current_console_mode == "ai":
            self.show_manual_panel()

    def refresh_manual_assignments(self, show_panel: bool = False):
        self.manual_panel.set_assignments(self.text_edit.build_manual_assignments())
        if show_panel:
            self.show_manual_panel()

    def apply_voice_assignments(self, assignments: List[dict], clear_existing: bool = True):
        current_mode = self.current_console_mode
        self.text_edit.apply_block_assignments(assignments, clear_existing=clear_existing)
        self.refresh_manual_assignments(show_panel=False)
        if current_mode == "manual":
            self.show_manual_panel()
        else:
            self.show_ai_panel()

    def apply_current_ai_assignments(self, clear_existing: bool = True) -> int:
        assignments = self.ai_panel.build_assignments_for_all_groups()
        if assignments:
            self.apply_voice_assignments(assignments, clear_existing=clear_existing)
        return len(assignments)

    def normalize_text_content(self):
        if not self.text_edit.toPlainText().strip():
            return

        self.text_edit.normalize_text_content()
        self.clear_ai_assignments(show_manual=False)
        self.refresh_manual_assignments(show_panel=False)
        self.show_ai_panel()
        InfoBar.success(
            title="规范化完成",
            content="已移除换行、黑色背景和额外字体格式，角色打标已保留。",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2500,
            parent=self,
        )

    def format_text_content(self):
        if not self.text_edit.toPlainText().strip():
            return

        reply = QMessageBox.question(
            self,
            "确认格式化",
            "格式化会移除文本里的控制标签内容，例如 <strong>、[breath]、<|endofprompt|> 等标记，但会保留普通正文和角色打标。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.text_edit.strip_markup_tokens()
        self.clear_ai_assignments(show_manual=False)
        self.refresh_manual_assignments(show_panel=False)
        self.show_ai_panel()
        InfoBar.success(
            title="格式化完成",
            content="已清理文本里的控制标签内容，并保留现有角色打标。",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2500,
            parent=self,
        )
