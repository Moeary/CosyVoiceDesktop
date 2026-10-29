import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QButtonGroup, QFileDialog, QHBoxLayout, QHeaderView, QTableWidgetItem, QWidget

from qfluentwidgets import (
    BodyLabel, CaptionLabel, ComboBox, FluentIcon, LineEdit, MessageBoxBase, PlainTextEdit, PushButton,
    RadioButton, SubtitleLabel, TableWidget, ToolButton,
)

GENDERS = ['男', '女', '群体', '未知']
AGES = ['儿童', '少年', '青年', '中年', '老年', '未知']


def _readonly(text: str, tooltip: str = '') -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    if tooltip:
        item.setToolTip(tooltip)
    return item


class RosterReviewDialog(MessageBoxBase):
    """分角跑完后先给用户看角色名单：可以直接改名/性别/年龄/语气后确认，也可以写意见让智能体重新分配。"""

    COLUMNS = ["角色", "别名", "性别", "年龄", "语气", "台词数", "示例台词"]

    def __init__(self, payload: dict, round_no: int = 1, parent=None):
        super().__init__(parent)
        self.payload = payload
        self.roster = payload.get('roster', [])
        self.action = 'approve'

        title = "确认角色名单" + (f"（第 {round_no} 版）" if round_no > 1 else "")
        self.titleLabel = SubtitleLabel(title, self)
        summary = (
            f"共 {payload.get('segments', 0)} 个片段：{len(self.roster)} 个角色，旁白 {payload.get('narration_lines', 0)} 段"
            + (f"，系统 {payload['system_lines']} 段" if payload.get('system_lines') else '')
            + (f"，{payload['unknown_lines']} 段未识别说话人" if payload.get('unknown_lines') else '')
            + "。可以直接修改表格里的角色名、性别、年龄、语气（改成已有角色名即合并）。"
        )
        self.summaryLabel = BodyLabel(summary, self)
        self.summaryLabel.setWordWrap(True)
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.summaryLabel)

        if payload.get('reply'):
            reply = CaptionLabel(f"上次意见的处理结果：{payload['reply']}", self)
            reply.setWordWrap(True)
            reply.setStyleSheet("color: #1d7a53;")
            self.viewLayout.addWidget(reply)

        self.table = TableWidget(self)
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setRowCount(len(self.roster))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        for column, width in enumerate([110, 120, 80, 80, 130, 60]):
            self.table.setColumnWidth(column, width)

        self.gender_boxes, self.age_boxes = [], []
        for row, character in enumerate(self.roster):
            self.table.setItem(row, 0, QTableWidgetItem(character['name']))
            aliases = '、'.join(character.get('aliases', []))
            self.table.setItem(row, 1, _readonly(aliases, aliases))
            for options, key, boxes, column in ((GENDERS, 'gender', self.gender_boxes, 2), (AGES, 'age', self.age_boxes, 3)):
                box = ComboBox(self.table)
                box.addItems(options)
                box.setCurrentText(character.get(key) or '未知')
                boxes.append(box)
                self.table.setCellWidget(row, column, box)
            style = QTableWidgetItem(character.get('voice_style', ''))
            style.setToolTip("配音时的语气指令，如 沉稳低缓、泼辣大嗓门")
            self.table.setItem(row, 4, style)
            self.table.setItem(row, 5, _readonly(str(character.get('lines', 0))))
            samples = character.get('samples', [])
            self.table.setItem(row, 6, _readonly(
                ' / '.join(f"#{s['id']} {s['text']}" for s in samples),
                '\n'.join(f"#{s['id']} {s['text']}" for s in samples),
            ))
        self.table.setMinimumHeight(min(420, 60 + 44 * max(len(self.roster), 3)))
        self.viewLayout.addWidget(self.table)

        issues = payload.get('issues') or []
        if issues:
            issue_label = CaptionLabel(
                "仍存疑的片段：" + '；'.join(f"#{i['id']} {i['problem']}「{i['text']}」" for i in issues[:5])
                + ('…' if len(issues) > 5 else ''), self)
            issue_label.setWordWrap(True)
            issue_label.setStyleSheet("color: #8f5b00;")
            self.viewLayout.addWidget(issue_label)

        self.feedbackEdit = PlainTextEdit(self)
        self.feedbackEdit.setPlaceholderText(
            "有异议就写在这里，智能体会按意见修订角色表并只重标受影响的片段，例如：\n"
            "#28 是赵德厚说的；蒙面人就是柳随风；漏了一个角色老张；王二婶说话泼辣、嗓门大"
        )
        self.feedbackEdit.setFixedHeight(90)
        self.feedbackEdit.textChanged.connect(self._update_buttons)
        self.viewLayout.addWidget(self.feedbackEdit)

        self.reviseButton = PushButton("按意见重新分配", self.buttonGroup)
        self.reviseButton.clicked.connect(self._revise)
        self.buttonLayout.insertWidget(1, self.reviseButton, 1, Qt.AlignVCenter)
        self.yesButton.setText("确认名单")
        self.cancelButton.setText("稍后再说")
        self.widget.setMinimumWidth(960)
        self._update_buttons()

    def _update_buttons(self):
        has_feedback = bool(self.feedbackEdit.toPlainText().strip())
        self.reviseButton.setEnabled(has_feedback)
        self.yesButton.setEnabled(not has_feedback)

    def _revise(self):
        self.action = 'revise'
        self.accept()

    def edits(self):
        edits = []
        for row, character in enumerate(self.roster):
            edit = {}
            name = (self.table.item(row, 0).text() or '').strip()
            if name and name != character['name']:
                edit['new_name'] = name
            if self.gender_boxes[row].currentText() != (character.get('gender') or '未知'):
                edit['gender'] = self.gender_boxes[row].currentText()
            if self.age_boxes[row].currentText() != (character.get('age') or '未知'):
                edit['age'] = self.age_boxes[row].currentText()
            style = (self.table.item(row, 4).text() or '').strip()
            if style != (character.get('voice_style') or ''):
                edit['voice_style'] = style
            if edit:
                edits.append({'name': character['name'], **edit})
        return edits

    def decision(self) -> dict:
        return {
            'action': self.action,
            'feedback': self.feedbackEdit.toPlainText().strip() if self.action == 'revise' else '',
            'edits': self.edits(),
        }


class RoleTableDialog(MessageBoxBase):
    """名单确认后选择角色表的参考音频来源。"""

    def __init__(self, character_count: int, has_voice_configs: bool, library_dir: str = '', parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel("生成角色表", self)
        tip = BodyLabel(
            f"名单已确认，共 {character_count} 个角色。角色表会为每个角色建一条语音配置（追加到语音设置，已有同名配置保留不动），"
            "有语气的角色默认使用指令控制模式。参考音频从哪里来？", self)
        tip.setWordWrap(True)
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(tip)

        self.group = QButtonGroup(self)
        self.manualRadio = RadioButton("生成角色表，参考音频和参考文本由我自己填写", self)
        self.libraryRadio = RadioButton("生成角色表，并从参考音频目录中自动匹配（会调用一次模型挑选说话人）", self)
        self.existingRadio = RadioButton("不生成角色表，映射到现有的音色配置", self)
        self.existingRadio.setEnabled(has_voice_configs)
        for radio in (self.manualRadio, self.libraryRadio, self.existingRadio):
            self.group.addButton(radio)
            self.viewLayout.addWidget(radio)

        dir_row = QWidget(self)
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(28, 0, 0, 0)
        self.dirEdit = LineEdit(dir_row)
        self.dirEdit.setPlaceholderText("参考音频目录：按说话人分文件夹，或带 .list 标注清单，同名 .txt/.lab 为参考文本")
        self.dirEdit.setText(library_dir)
        browse = ToolButton(FluentIcon.FOLDER, dir_row)
        browse.clicked.connect(self._browse)
        dir_layout.addWidget(self.dirEdit, 1)
        dir_layout.addWidget(browse)
        self.viewLayout.addWidget(dir_row)

        self.errorLabel = CaptionLabel('', self)
        self.errorLabel.setStyleSheet("color: #b13a3a;")
        self.viewLayout.addWidget(self.errorLabel)

        (self.libraryRadio if library_dir else self.manualRadio).setChecked(True)
        self.yesButton.setText("生成")
        self.cancelButton.setText("跳过")
        self.widget.setMinimumWidth(640)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "选择参考音频目录", self.dirEdit.text())
        if path:
            self.dirEdit.setText(path)
            self.libraryRadio.setChecked(True)

    def choice(self) -> str:
        if self.libraryRadio.isChecked():
            return 'library'
        if self.existingRadio.isChecked():
            return 'existing'
        return 'manual'

    def library_dir(self) -> str:
        return self.dirEdit.text().strip()

    def validate(self) -> bool:
        if self.choice() == 'library' and not os.path.isdir(self.library_dir()):
            self.errorLabel.setText("参考音频目录不存在")
            return False
        return True
