from PyQt5.QtWidgets import QHeaderView, QTableWidgetItem

from qfluentwidgets import BodyLabel, ComboBox, MessageBoxBase, SubtitleLabel, TableWidget


class RoleAssignmentDialog(MessageBoxBase):
    """AI 角色分配确认对话框"""

    def __init__(self, assignments, voice_configs, parent=None):
        super().__init__(parent)
        self.assignments = assignments
        self.voice_configs = voice_configs
        self.combo_boxes = []

        self.titleLabel = SubtitleLabel("确认角色分配", self)
        self.tipLabel = BodyLabel("模型已返回建议结果，你可以在应用前手动调整每一段对应的角色。", self)
        self.tipLabel.setWordWrap(True)

        self.table = TableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["段落", "文本内容", "角色", "置信度", "说明"])
        self.table.verticalHeader().setVisible(False)
        self.table.setRowCount(len(assignments))

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(60)
        self.table.setColumnWidth(0, 70)
        self.table.setColumnWidth(1, 360)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnWidth(3, 80)

        speaker_names = list(voice_configs.keys())
        for row, item in enumerate(assignments):
            index_item = QTableWidgetItem(str(item.get('index', row + 1)))
            text_item = QTableWidgetItem(item.get('text', ''))
            text_item.setToolTip(item.get('text', ''))

            combo = ComboBox(self.table)
            combo.addItems(speaker_names)
            speaker = item.get('speaker', '')
            if speaker in speaker_names:
                combo.setCurrentText(speaker)
            self.combo_boxes.append(combo)

            confidence = item.get('confidence')
            confidence_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "-"
            confidence_item = QTableWidgetItem(confidence_text)
            reason_item = QTableWidgetItem(item.get('reason', '') or '-')
            reason_item.setToolTip(item.get('reason', '') or '')

            self.table.setItem(row, 0, index_item)
            self.table.setItem(row, 1, text_item)
            self.table.setCellWidget(row, 2, combo)
            self.table.setItem(row, 3, confidence_item)
            self.table.setItem(row, 4, reason_item)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.tipLabel)
        self.viewLayout.addWidget(self.table)

        self.widget.setMinimumWidth(900)
        self.table.setMinimumHeight(420)

        self.yesButton.setText("应用标签")
        self.cancelButton.setText("取消")

    def get_assignments(self):
        results = []
        for row, item in enumerate(self.assignments):
            results.append({
                'index': item.get('index', row + 1),
                'speaker': self.combo_boxes[row].currentText(),
                'confidence': item.get('confidence'),
                'reason': item.get('reason', ''),
                'text': item.get('text', ''),
                'current_speaker': item.get('current_speaker', ''),
            })
        return results
