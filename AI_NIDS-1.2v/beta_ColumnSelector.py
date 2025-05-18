# beta_ColumnSelector.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QListWidget, QMessageBox

class ColumnSelector(QDialog):
    def __init__(self, column_names, dataset, parent=None):
        super(ColumnSelector, self).__init__(parent)
        self.dataset = dataset  # 데이터셋을 인자로 추가
        self.setWindowTitle('Select column')
        self.layout = QVBoxLayout(self)
        self.list_widget = QListWidget(self)
        self.list_widget.addItems(column_names)
        self.layout.addWidget(QLabel('Select a value to normalize:'))
        self.layout.addWidget(self.list_widget)
        self.select_button = QPushButton('Select', self)
        self.select_button.clicked.connect(self.select_and_normalize)
        self.layout.addWidget(self.select_button)
        self.selected_column_name = None

    def select_and_normalize(self):
        selected_item = self.list_widget.currentItem()
        if selected_item:  # 선택된 항목이 있는지 확인
            self.selected_column_name = selected_item.text()
            self.normalize_columns(self.selected_column_name)
            self.accept()
        else:
            # 사용자가 열을 선택하지 않았을 때 오류 메시지 표시
            QMessageBox.warning(self, "Error", "You must select a column.")
            self.reject()

    def normalize_columns(self, column_name):
        if self.dataset is None:
            raise ValueError("Dataset has not been loaded.")
        if column_name in self.dataset.columns:
            self.dataset[column_name] = self.normalize(self.dataset[column_name])

    @staticmethod
    def normalize(column):
        # 열 정규화 로직
        return (column - column.min()) / (column.max() - column.min())

    def get_selected_column(self):
        return self.selected_column_name