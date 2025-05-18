#beta_main.py
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidget, QTableWidgetItem, QInputDialog, \
    QProgressBar, QVBoxLayout, QDialog, QWidget,QMessageBox,QLabel, QComboBox, QPushButton
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics import confusion_matrix
from mplwidget import mplwidget  # 사용자 정의 matplotlib 위젯
from beta_data_preprocessor import DataPreprocessor
from beta_model_trainer import ModelTrainer
from beta_model_evaluator import ModelEvaluator
from math import floor
from beta_ranking import RankingManager
from beta_thread import TrainingThread
from beta_callback import CustomCallback
from keras.models import Sequential, load_model
from joblib import dump, load
from PyQt5.QtWidgets import QFileDialog



class WindowClass(QMainWindow):
    def __init__(self):
        super(WindowClass, self).__init__()
        loadUi("beta_ui.ui", self)
        self.ranking_data = []  # 랭킹 데이터를 저장할 리스트 초기화


        self.selected_algorithm_label = None
        self.selected_algorithm_label = self.findChild(QLabel, 'selected_algorithm_label')
        self.Algorithm_select_button.clicked.connect(self.show_algorithm_selection)

        # 버튼들을 각각의 메서드에 연결
        self.Preprocessing_select_button.clicked.connect(self.show_preprocessing_selection)

        self.data_preprocessor = None

        self.save_model_button.clicked.connect(self.save_model)
        self.load_model_button.clicked.connect(self.load_model)

        self.tableWidget = QTableWidget(self)
        self.dataset_scrollarea.setWidget(self.tableWidget)

        self.Database_select_button.clicked.connect(self.select_dataset)
        self.train_button.clicked.connect(self.train_model)
        self.hidden_layer_select.currentIndexChanged.connect(self.enable_start_train_button)
        self.output_layer_select.currentIndexChanged.connect(self.enable_start_train_button)

        # 프로그레스 바 및 테이블 위젯 초기화
        self.progress_bar = self.findChild(QProgressBar, 'progressBar')
        self.progress_bar.setValue(1)

        # mplwidget을 플레이스홀더 위젯의 위치에 배치합니다.
        self.mplwidget = mplwidget(self)
        self._setup_mplwidget()
        self.data_preprocessor = None
        self.is_trained = False  # 모델이 훈련되었는지 여부를 나타내는 변수
        self.training_thread = None  # TrainingThread 인스턴스 변수 추가

        # RankingManager 인스턴스 생성
        self.ranking_manager = RankingManager(self.RankingTable)

        # save_ranking_button에 대한 설정과 메서드 연결
        self.save_ranking_button = self.findChild(QPushButton, 'save_ranking_button')
        self.save_ranking_button.clicked.connect(self.save_ranking)

        # select_ranking 버튼에 대한 설정과 메서드 연결
        self.select_ranking_button = self.findChild(QPushButton, 'select_ranking_button')
        self.select_ranking_button.clicked.connect(self.select_ranking)

    def save_model(self):
        # 모델 저장 대화상자
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Model Files (*.h5 *.joblib)")
        if file_path:
            if isinstance(self.trained_model, Sequential):
                # Keras 모델 저장
                self.trained_model.save(file_path)
            else:
                # Scikit-learn 모델 저장
                dump(self.trained_model, file_path)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Model Files (*.h5 *.joblib)")
        if file_path:
            try:
                if file_path.endswith('.h5'):
                    # Keras 모델 불러오기
                    self.trained_model = load_model(file_path)
                    self.is_trained = True
                else:
                    # Scikit-learn 모델 불러오기
                    self.trained_model = load(file_path)
                    self.is_trained = True

                QMessageBox.information(self, "Load Successful", "Model loaded successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Load Failed", f"Failed to load model: {str(e)}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Model Files (*.h5 *.joblib)")
        if file_path:
            try:
                if file_path.endswith('.h5'):
                    # Keras 모델 불러오기
                    self.trained_model = load_model(file_path)
                    self.is_trained = True

                    # 모델 구조 출력
                    print(self.trained_model.summary())

                    QMessageBox.information(self, "Load Successful", "Model loaded successfully.")
                else:
                    # Scikit-learn 모델 불러오기
                    self.trained_model = load(file_path)
                    self.is_trained = True

                    QMessageBox.information(self, "Load Successful", "Model loaded successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Load Failed", f"Failed to load model: {str(e)}")

    def continue_training(self):
        # 새 데이터셋 선택 및 로드
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Dataset for Additional Training", "", "CSV Files (*.csv);;All Files (*)")
        if filePath:
            # 새 DataPreprocessor 인스턴스 생성 및 데이터 전처리
            additional_data_preprocessor = DataPreprocessor(filePath)
            additional_data_preprocessor.preprocess_data()
            
            # 인코딩 방법 선택 (원-핫 인코딩 또는 레이블 인코딩)
            # 예시로는 원-핫 인코딩을 사용
            additional_data_preprocessor.one_hot_encode_string_columns()

            # 학습 데이터 및 레이블 분리
            x_train_additional = additional_data_preprocessor.x
            y_train_additional = additional_data_preprocessor.y

            # ModelTrainer 인스턴스 생성 및 추가 학습 수행
            model_trainer = ModelTrainer(self.data_preprocessor.x, self.data_preprocessor.y)
            self.trained_model, accuracy, loss = model_trainer.continue_training(
                self.trained_model, 
                x_train_additional, 
                y_train_additional, 
                epochs=10,  # 원하는 에포크 수를 설정합니다.
                batch_size=50  # 원하는 배치 크기를 설정합니다.
            )

            # 결과 메시지 표시
            QMessageBox.information(self, "Training Complete", "Additional training completed.")


    def start_new_training(self):
        # 새 모델 훈련 시작 로직
        self.disable_ui(True)
        self.x_test_scrollarea.setEnabled(False)
        self.statusBar().showMessage("모델 학습 중...")

        self.training_thread = TrainingThread(
            model_trainer=ModelTrainer(self.data_preprocessor.x, self.data_preprocessor.y),
            x=self.data_preprocessor.x,
            y=self.data_preprocessor.y,
            activation_function=self.hidden_layer_select.currentText(),
            output_layer_type=self.output_layer_select.currentText(),
            progress_bar=self.progress_bar,
            mplwidget=self.mplwidget
        )
        self.training_thread.training_completed.connect(self.on_training_completed)
        self.training_thread.start()


    def show_algorithm_selection(self):
        algorithms = ["Random Forest", "GaussianNB", "Decision Tree"]
        selected_algorithm, okPressed = QInputDialog.getItem(self, "Select Algorithm", "Algorithm:", algorithms, 0, False)

        if okPressed and selected_algorithm:
            self.selected_algorithm_label.setText(selected_algorithm)
            model_trainer = ModelTrainer(self.data_preprocessor.x, self.data_preprocessor.y)

            if selected_algorithm == "Random Forest":
                output_layer_type = 'random-forest'
            elif selected_algorithm == "GaussianNB":
                output_layer_type = 'gaussian-nb'
            elif selected_algorithm == "Decision Tree":
                output_layer_type = 'decision-tree'

            self.trained_model, self.x_test, self.y_test, self.y_proba, _ = model_trainer.train_model_with_activation(
                output_layer_type=output_layer_type, 
                callbacks=[CustomCallback(progress_bar=self.progress_bar, mplwidget=self.mplwidget)]
        )


    def show_algorithm_label(self, message):
        self.selected_algorithm_label.setText(message)



    def _setup_mplwidget(self):
        # 이미 만들어진 mplwidget 인스턴스를 레이아웃에 추가합니다.
        placeholder = self.findChild(QWidget, 'loss_accuracy')
        layout = QVBoxLayout()
        layout.addWidget(self.mplwidget)
        placeholder.setLayout(layout)

    def show_preprocessing_selection(self):
        preprocessings = ["one-hot encoding", "label-encoding"]
        selected_preprocessing, okPressed = QInputDialog.getItem(self, "Select preprocessing", "preprocessing:", preprocessings, 0,
                                                             False)

        if okPressed and selected_preprocessing:
            self.selected_preprocessing_label.setText(selected_preprocessing)
            if selected_preprocessing == "one-hot encoding":
                if self.data_preprocessor:
                    self.data_preprocessor.one_hot_encode_string_columns()
            elif selected_preprocessing == "label-encoding":
                if self.data_preprocessor:
                    self.data_preprocessor.label_encode_string_columns()

        self.x_test = self.data_preprocessor.x
        # x_train 데이터를 테이블에 표시
        self.fillXTestTable()

    def _apply_preprocessing(self, algorithm):
        # 선택된 전처리 알고리즘 적용
        if self.data_preprocessor:
            if algorithm == "one-hot encoding":
                self.data_preprocessor.one_hot_encode_string_columns()
            elif algorithm == "label-encoding":
                self.data_preprocessor.label_encode_string_columns()
            self.x_test = self.data_preprocessor.x
            self.fillXTestTable()

    def enable_start_train_button(self):
        # 코딩 방법과 출력층 타입이 선택되었을 때 Start Train 버튼 활성화
        coding_method = self.hidden_layer_select.currentText()
        output_layer_type = self.output_layer_select.currentText()
        if coding_method and output_layer_type:
            self.train_button.setEnabled(True)
        else:
            self.train_button.setEnabled(False)

    def select_dataset(self):
        # 데이터셋 선택 및 로드
        filePath, _ = QFileDialog.getOpenFileName(self, "데이터셋 선택", "", "CSV 파일 (*.csv);;모든 파일 (*)")
        if filePath:
            self.data_preprocessor = DataPreprocessor(filePath)
            self.data_preprocessor.preprocess_data()
            self.dataset = self.data_preprocessor.dataset
            self.dataset_filename_label.setText(filePath.split('/')[-1])
            self.dataset_path_label.setText(filePath)
            self.progress_bar.setValue(10)
            self.fillTable()
            self.show_preprocessing_selection()

    def fillTable(self):
        # 데이터 테이블 채우기
        self.tableWidget.setRowCount(10)
        self.tableWidget.setColumnCount(self.dataset.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(self.dataset.columns)
        for i in range(10):
            for j in range(self.dataset.shape[1]):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.dataset.iat[i, j])))

    def disable_ui(self, disable):
        """UI 요소들 활성화/비활성화"""
        self.Database_select_button.setDisabled(disable)
        self.Preprocessing_select_button.setDisabled(disable)
        self.train_button.setDisabled(disable)

    def train_model(self):
        self.progress_bar.setValue(0)
        if not self.is_trained or (self.is_trained and self.training_thread is None):
            # 모델이 훈련되지 않았거나, 훈련 중인 스레드가 없는 경우에만 실행
            # 모델 학습 시작
            self.disable_ui(True)
            self.x_test_scrollarea.setEnabled(False)
            self.statusBar().showMessage("모델 학습 중...")

        self.training_thread = TrainingThread(
            model_trainer=ModelTrainer(self.data_preprocessor.x, self.data_preprocessor.y),
            x=self.data_preprocessor.x,
            y=self.data_preprocessor.y,
            activation_function=self.hidden_layer_select.currentText(),
            output_layer_type=self.output_layer_select.currentText(),
            progress_bar=self.progress_bar,
            mplwidget=self.mplwidget
        )
        self.training_thread.training_completed.connect(self.on_training_completed)
        self.training_thread.start()


    def on_training_completed(self):
        # 학습 완료 후 실행
        self._update_model_data()
        self._evaluate_model()
        self.progress_bar.setValue(100)
        self.showGraphDialog()
        self.disable_ui(False)
        self.x_test_scrollarea.setEnabled(True)
        self.statusBar().clearMessage()
        self.is_trained = True  # 모델이 훈련되었음을 표시
        self.training_thread = None  # 훈련 스레드 초기화

    def _update_model_data(self):
        # 훈련된 모델 데이터 업데이트
        self.trained_model = self.training_thread.trained_model
        self.x_test = self.training_thread.x_test
        self.y_test = self.training_thread.y_test
        self.accuracy_data = self.training_thread.accuracy_data
        self.loss_data = self.training_thread.loss_data

    def _evaluate_model(self):
        # 모델 평가
        evaluator = ModelEvaluator(self.trained_model)
        self.accuracy = evaluator.evaluate_model(self.x_test, self.y_test)
        self.f1 = evaluator.calculate_f1_score(self.x_test, self.y_test)
        self.accuracy_f1score.setText(f"정확도: {self.accuracy} / F1 점수: {self.f1}")
        # 모델 평가 결과를 랭킹 매니저에 추가
        self.ranking_manager.add_new_ranking(
            self.dataset_filename_label.text(),
            self.selected_algorithm_label.text(),
            self.selected_preprocessing_label.text(),
            self.accuracy,
            self.f1,
            self.hidden_layer_select.currentText(),
            self.output_layer_select.currentText()
        )

    def showGraphDialog(self):
        # 그래프 및 혼동 행렬 다이얼로그 표시
        dialog = QDialog(self)
        dialog.setWindowTitle("Accuracy, loss, and confusion matrix")
        fig, axs = plt.subplots(3, 1, figsize=(5, 8))
        canvas = FigureCanvas(fig)
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        dialog.setLayout(layout)
        self._plot_graphs(axs)
        dialog.exec_()

    def _plot_graphs(self, axs):
        # 그래프 플로팅
        epochs = range(1, len(self.accuracy_data) + 1)
        axs[0].plot(epochs, self.accuracy_data)
        axs[0].set_title('accuracy')
        axs[1].plot(epochs, self.loss_data)
        axs[1].set_title('loss')
        self._plot_confusion_matrix(axs[2])

    def _plot_confusion_matrix(self, ax):
        # 혼동 행렬 플로팅
        y_pred_prob = self.trained_model.predict(self.x_test)
        y_pred = (y_pred_prob > 0.5).astype(int) if y_pred_prob.shape[1] == 1 else y_pred_prob.argmax(axis=1)
        y_true = self.y_test.argmax(axis=1) if len(self.y_test.shape) > 1 and self.y_test.shape[1] > 1 else self.y_test
        conf_matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax)
        ax.set_title('confusion_matrix')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    def fillXTestTable(self):
        # x_test 테이블 채우기
        x_test_df = pd.DataFrame(self.x_test)
        self.x_test_table = QTableWidget()
        self.x_test_table.setRowCount(10)
        self.x_test_table.setColumnCount(x_test_df.shape[1])
        for i in range(10):
            for j in range(x_test_df.shape[1]):
                self.x_test_table.setItem(i, j, QTableWidgetItem(str(x_test_df.iat[i, j])))
        self.x_test_scrollarea.setWidget(self.x_test_table)
        self.x_test_table.itemSelectionChanged.connect(self.showSelectedRow)

    def showSelectedRow(self):
        # 선택된 행 표시
        selected_row = self.x_test_table.currentRow()
        selected_data = [self.x_test_table.item(selected_row, col).text() for col in range(self.x_test_table.columnCount())]
        corresponding_y_test = self.y_test[selected_row]
        self.y_test_textBrowser.setText(str(corresponding_y_test))
        selected_row_table = QTableWidget()
        selected_row_table.setRowCount(1)
        selected_row_table.setColumnCount(len(selected_data))
        for j, data in enumerate(selected_data):
            selected_row_table.setItem(0, j, QTableWidgetItem(str(data)))
        self.x_test_scrollarea_2.setWidget(selected_row_table)

    def update_ranking_table(self):
        # 정확도에 따라 랭킹 데이터 정렬
        sorted_data = sorted(self.ranking_data, key=lambda x: x['Accuracy'], reverse=True)

        # RankingTable 위젯 설정
        self.RankingTable.setRowCount(len(sorted_data))
        self.RankingTable.setColumnCount(7)
        self.RankingTable.setHorizontalHeaderLabels(['Dataset', 'algorithm', 'preprocessing', 'Accuracy', 'F1 Score', 'HiddenLayer', 'OutputLayer'])

        # 정렬된 데이터로 RankingTable 채우기
        for row, entry in enumerate(sorted_data):
            self.RankingTable.setItem(row, 0, QTableWidgetItem(entry['Dataset']))
            self.RankingTable.setItem(row, 1, QTableWidgetItem(entry['algorithm']))
            self.RankingTable.setItem(row, 2, QTableWidgetItem(entry['preprocessing']))
            self.RankingTable.setItem(row, 3, QTableWidgetItem(f"{floor(entry['Accuracy'] * 100) / 100:.2f}"))
            self.RankingTable.setItem(row, 4, QTableWidgetItem(f"{floor(entry['F1Score'] * 100) / 100:.2f}"))
            self.RankingTable.setItem(row, 5, QTableWidgetItem(entry['HiddenLayer']))
            self.RankingTable.setItem(row, 6, QTableWidgetItem(entry['OutputLayer']))

        # RankingTable의 크기를 내용에 맞게 조정
        self.RankingTable.resizeColumnsToContents()

    def save_ranking(self):
        # 파일 저장 대화 상자를 통해 사용자가 파일을 저장할 경로를 선택하도록 합니다.
        file_path, _ = QFileDialog.getSaveFileName(self, "랭킹 저장", "", "CSV 파일 (*.csv);;모든 파일 (*)")
        if file_path:
            # RankingManager의 save_ranking 메서드를 호출
            self.ranking_manager.save_ranking(file_path)
            QMessageBox.information(self, "저장 완료", "랭킹이 성공적으로 저장되었습니다.")

    def select_ranking(self):
        # 파일 선택 대화 상자를 통해 사용자가 랭킹 파일을 선택하도록 합니다.
        file_path, _ = QFileDialog.getOpenFileName(self, "랭킹 파일 선택", "", "CSV 파일 (*.csv);;모든 파일 (*)")
        if file_path:
            # 선택된 파일의 랭킹 데이터를 읽어옵니다.
            ranking_data = pd.read_csv(file_path)

            # RankingManager의 랭킹 데이터와 테이블을 업데이트합니다.
            self.ranking_manager.ranking_data = ranking_data.to_dict('records')
            self.ranking_manager.update_ranking_table()
            QMessageBox.information(self, "랭킹 로드", "랭킹이 성공적으로 로드되었습니다.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()