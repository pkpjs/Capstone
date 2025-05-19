#beta_data_preprocessor.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from beta_ColumnSelector import ColumnSelector

# DataPreprocessor 클래스 정의
class DataPreprocessor:
    def __init__(self, dataset_path):
        # 클래스 초기화 함수입니다. 여기에서 데이터셋을 불러옵니다.
        self.dataset = pd.read_csv(dataset_path)  # CSV 파일로부터 데이터셋을 불러옵니다.
        self.x = None  # 특징(feature) 데이터를 저장할 변수입니다.
        self.y = None  # 레이블(label) 데이터를 저장할 변수입니다.
        self.encoding_type = None

    def preprocess_data(self):
        # 데이터 전처리를 시작하는 함수입니다.
        self.select_column_for_normalization() # 'normal.' 컬럼 선택 메서드 호출
        self._split_features_labels() # 특성과 레이블을 분리하는 메서드 호출.

    def select_column_for_normalization(self, parent=None):
        if 'normal.' not in self.dataset.columns:
            unique_values_in_last_column = self.dataset.iloc[:, -1].unique()
            unique_values_in_last_column = [str(value) for value in unique_values_in_last_column]

            column_selector_dialog = ColumnSelector(unique_values_in_last_column, self.dataset, parent)
            if column_selector_dialog.exec_():
                selected_value = column_selector_dialog.get_selected_column()
                self._rename_column_to_normal(selected_value)
            else:
                pass
        else:
            self._rename_column_to_normal('normal.')

    def _rename_column_to_normal(self, selected_value):
        # 사용자가 선택한 값에 해당하는 컬럼을 'normal.'로 변경
        self.dataset.rename(columns={self.dataset.columns[-1]: 'normal.'}, inplace=True)
        self.dataset.loc[self.dataset.iloc[:, -1] != selected_value, 'normal.'] = 'attack'
        self.dataset.loc[self.dataset['normal.'] == selected_value, 'normal.'] = 'normal.'

    def _split_features_labels(self):
        # 데이터셋을 특징(X)과 레이블(y)로 분리합니다.
        self.x = self.dataset.iloc[:, :self.dataset.shape[1] - 1].values  # 마지막 열을 제외한 모든 열을 특징으로 설정합니다.
        self.y = self.dataset.iloc[:, self.dataset.shape[1] - 1].values   # 마지막 열을 레이블로 설정합니다.
        self.y = LabelEncoder().fit_transform(self.y)  # 레이블을 숫자로 변환합니다.

    def set_encoding_type(self, encoding_type):
        # 인코딩 타입을 설정하는 메소드입니다.
        self.encoding_type = encoding_type

    def label_encode_string_columns(self):
        # 문자열 타입의 컬럼을 레이블 인코딩합니다.
        string_columns_idx = [idx for idx, dtype in enumerate(self.x[0]) if isinstance(dtype, str)]
        label_encoders = [LabelEncoder() for _ in string_columns_idx]

        for i, idx in enumerate(string_columns_idx):
            self.x[:, idx] = label_encoders[i].fit_transform(self.x[:, idx])

    def one_hot_encode_string_columns(self):
        # 문자열 타입의 컬럼을 원-핫 인코딩합니다.
        string_columns_idx = [idx for idx, dtype in enumerate(self.x[0]) if isinstance(dtype, str)]

        column_transformers = []
        for i, idx in enumerate(string_columns_idx):
            column_transformers.append(("encoder_" + str(idx), OneHotEncoder(), [idx]))

        ct = ColumnTransformer(column_transformers, remainder='passthrough')
        self.x = ct.fit_transform(self.x)