#beta_model_trainer.py
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from keras.callbacks import History
from keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier



# CustomCallback 클래스를 import합니다. (실제 파일 위치에 따라 경로가 달라질 수 있음)
from beta_callback import CustomCallback

# ModelTrainer 클래스는 모델 훈련을 관리합니다.
class ModelTrainer:
    def __init__(self, x, y):
        # 초기화 함수입니다. 특징(X)과 레이블(y) 데이터를 입력받습니다.
        self.x = x
        self.y = y
        self.model = None  # 훈련할 모델을 저장할 변수입니다.

    def train_model_with_activation(self, activation_function='relu', output_layer_type='sigmoid', callbacks=None,
                                    progress_bar=None, mplwidget=None):
        if callbacks is None:
            callbacks = []

        if output_layer_type.lower() == 'random-forest':
            print("Training random-forest model")
            # 랜덤 포레스트 모델 훈련
            self.model = RandomForestClassifier(random_state=0)
            self.model.fit(self.x, self.y)

            # 랜덤 포레스트는 predict_proba를 지원하므로 각 클래스의 확률을 반환합니다.
            # 여기서는 클래스 1의 확률을 사용합니다.
            y_proba = self.model.predict_proba(self.x)
            # Random Forest는 loss가 없음
            return self.model, self.x, self.y, y_proba, None

        elif output_layer_type.lower() == 'decision-tree':
            print("Training decision-tree model")
            # Decision Tree 모델 훈련
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0)
            self.model = DecisionTreeClassifier()
            self.model.fit(x_train, y_train)
            # Decision Tree는 loss가 없음
            return self.model, x_test, y_test, None, None

        elif output_layer_type.lower() == 'gaussian-nb':
            print("Training Gaussian Naive Bayes model")
            # Gaussian Naive Bayes 모델 훈련
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0)
            self.model = GaussianNB()
            self.model.fit(x_train, y_train)
            # Gaussian Naive Bayes는 loss가 없음
            return self.model, x_test, y_test, None, None

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0)
        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(seed)

        x_train = np.asarray(x_train).astype('float32')
        x_test = np.asarray(x_test).astype('float32')
        y_train = np.asarray(y_train).astype('float32')
        y_test = np.asarray(y_test).astype('float32')

        self.model = Sequential()
        self.model.add(Dense(30, input_dim=self.x.shape[1], activation=activation_function))


        if output_layer_type == 'sigmoid':
            self.model.add(Dense(1, activation='sigmoid'))
        elif output_layer_type == 'softmax':
            if len(self.y.shape) == 1:
                self.model.add(Dense(1, activation='softmax'))
            else:
                self.model.add(Dense(self.y.shape[1], activation='softmax'))
        elif output_layer_type == 'linear':
            self.model.add(Dense(1, activation='linear'))
        else:
            raise ValueError('Invalid output_layer_type')

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = History()
        if callbacks:
            callbacks.append(history)
        else:
            callbacks = [history]


        # CustomCallback을 생성하고, 진행 상태를 업데이트할 위젯을 전달합니다.
        custom_callback = CustomCallback(progress_bar=progress_bar, mplwidget=mplwidget)

        # 모델 훈련 시, CustomCallback을 콜백으로 추가합니다.
        self.model.fit(x_train, y_train, epochs=10, batch_size=50, callbacks=callbacks + [custom_callback])

        return self.model, x_test, y_test, history.history['accuracy'], history.history['loss']
    

    def continue_training(self, model: Model, x_train, y_train, epochs=10, batch_size=50, callbacks=None):
        """
        이미 훈련된 Keras 모델에 대해 추가 학습을 수행합니다.
        :param model: 훈련할 Keras 모델
        :param x_train: 학습 데이터
        :param y_train: 학습 레이블
        :param epochs: 학습 에포크 수
        :param batch_size: 배치 크기
        :param callbacks: 콜백 리스트
        :return: 훈련된 모델, 학습 정확도 및 손실 기록
        """
        if callbacks is None:
            callbacks = []

        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

        return model, history.history['accuracy'], history.history['loss']
