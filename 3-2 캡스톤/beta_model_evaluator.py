#beta_model_evaluator.py
from sklearn.metrics import f1_score

# ModelEvaluator 클래스를 정의합니다. 이 클래스는 모델의 평가를 담당합니다.
class ModelEvaluator:
    def __init__(self, model):
        # 클래스의 초기화 함수입니다. 모델 객체를 입력받아 멤버 변수에 저장합니다.
        self.model = model  # 평가할 모델을 저장합니다.

    def evaluate_model(self, x_test, y_test):
        # 모델을 평가하는 메소드입니다. 테스트 데이터를 사용하여 모델의 성능을 평가합니다.
        return self.model.evaluate(x_test, y_test)[1]  # 모델의 evaluate 메소드를 사용하여 정확도를 반환합니다.

    def calculate_f1_score(self, x_test, y_test):
        # F1 점수를 계산하는 메소드입니다.
        y_pred = self.model.predict(x_test)  # 모델을 사용하여 예측을 수행합니다.
        y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]  # 예측값을 0.5 기준으로 이진화합니다.
        return f1_score(y_test, y_pred)  # 실제 레이블과 예측 레이블을 비교하여 F1 점수를 계산합니다.