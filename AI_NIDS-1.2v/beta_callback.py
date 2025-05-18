# beta_callback.py
from keras.callbacks import Callback
from PyQt5.QtCore import pyqtSignal, QObject

class CustomCallback(Callback, QObject):
    update_progress_signal = pyqtSignal(int)
    update_graph_signal = pyqtSignal(list, list)  # 그래프 업데이트 신호

    def __init__(self, progress_bar, mplwidget):
        Callback.__init__(self)
        QObject.__init__(self)
        self.progress_bar = progress_bar
        self.mplwidget = mplwidget
        self.update_progress_signal.connect(self.progress_bar.setValue)
        self.update_graph_signal.connect(self.mplwidget.update_graph)  # 그래프 업데이트 연결

        self.loss_data = []  # 손실 데이터 초기화
        self.acc_data = []  # 정확도 데이터 초기화
        self.total_batches = 0  # 총 배치 수
        self.processed_batches = 0  # 처리된 배치 수
        self.current_epoch = 0  # 현재 에포크 번호를 저장할 속성

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch  # 현재 에포크 번호를 저장
        self.processed_batches = 0  # 처리된 배치 수를 초기화
        self.total_batches = self.params['steps']  # 총 배치 수를 저장

    def on_batch_end(self, batch, logs=None):
        self.processed_batches += 1  # 처리된 배치 수를 증가
        progress_value = 100 * (self.current_epoch + self.processed_batches / self.total_batches) / self.params[
            'epochs']
        self.update_progress_signal.emit(int(progress_value))  # 프로그레스 바 업데이트

    def on_epoch_end(self, epoch, logs=None):
        # 프로그레스 바 완전히 채우기
        progress_value = 100 * (epoch + 1) / self.params['epochs']
        self.update_progress_signal.emit(int(progress_value))
