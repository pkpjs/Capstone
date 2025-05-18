#beta_thread.py
from PyQt5.QtCore import QThread, pyqtSignal

# TrainingThread 클래스는 QThread를 상속받아 백그라운드에서 모델 훈련을 관리합니다.
class TrainingThread(QThread):
    # 훈련이 완료되었을 때 발생하는 신호입니다.
    training_completed = pyqtSignal()

    def __init__(self, model_trainer, x, y, activation_function, output_layer_type, progress_bar, mplwidget):
        # 초기화 함수입니다. 모델 훈련에 필요한 매개변수를 입력받습니다.
        QThread.__init__(self)
        self.model_trainer = model_trainer  # 모델 훈련 객체입니다.
        self.x = x  # 특징 데이터입니다.
        self.y = y  # 레이블 데이터입니다.
        self.activation_function = activation_function  # 활성화 함수입니다.
        self.output_layer_type = output_layer_type  # 출력 계층 유형입니다.
        self.progress_bar = progress_bar  # 프로그레스 바 위젯입니다.
        self.mplwidget = mplwidget  # Matplotlib 위젯입니다.

    def run(self):
        # 스레드의 주 실행 함수입니다. 모델 훈련 과정을 실행합니다.
        self.trained_model, self.x_test, self.y_test, self.accuracy_data, self.loss_data = \
            self.model_trainer.train_model_with_activation(
                activation_function=self.activation_function,
                output_layer_type=self.output_layer_type,
                progress_bar=self.progress_bar,
                mplwidget=self.mplwidget
            )
        # 훈련이 완료되면 신호를 발생시킵니다.
        self.training_completed.emit()