#beta_ranking.py
from PyQt5.QtWidgets import QTableWidgetItem
from math import floor
import pandas as pd

class RankingManager:
    def __init__(self, ranking_table_widget):
        self.ranking_data = []
        self.RankingTable = ranking_table_widget

    def add_new_ranking(self, dataset_name, algorithm, preprocessing, accuracy, f1_score, hidden_layer, output_layer):
        # 새 랭킹 데이터 추가
        self.ranking_data.append({
            'Dataset': dataset_name,
            'algorithm': algorithm,
            'preprocessing': preprocessing,
            'Accuracy': accuracy,
            'F1Score': f1_score,
            'HiddenLayer': hidden_layer,
            'OutputLayer': output_layer
        })
        # 테이블 업데이트
        self.update_ranking_table()

    def update_ranking_table(self):
        # 정렬된 랭킹 데이터로 테이블 업데이트
        sorted_data = sorted(self.ranking_data, key=lambda x: x['Accuracy'], reverse=True)
        self.RankingTable.setRowCount(len(sorted_data))
        self.RankingTable.setColumnCount(7)
        self.RankingTable.setHorizontalHeaderLabels(['Dataset', 'algorithm', 'preprocessing', 'Accuracy', 'F1 Score', 'HiddenLayer', 'OutputLayer'])

        for row, entry in enumerate(sorted_data):
            self.RankingTable.setItem(row, 0, QTableWidgetItem(str(entry['Dataset'])))
            self.RankingTable.setItem(row, 1, QTableWidgetItem(str(entry['algorithm'])))
            self.RankingTable.setItem(row, 2, QTableWidgetItem(str(entry['preprocessing'])))
            self.RankingTable.setItem(row, 3, QTableWidgetItem(f"{floor(entry['Accuracy'] * 100) / 100:.2f}"))
            self.RankingTable.setItem(row, 4, QTableWidgetItem(f"{floor(entry['F1Score'] * 100) / 100:.2f}"))
            self.RankingTable.setItem(row, 5, QTableWidgetItem(str(entry['HiddenLayer'])))
            self.RankingTable.setItem(row, 6, QTableWidgetItem(str(entry['OutputLayer'])))

    def save_ranking(self, file_path):
        # 랭킹 데이터를 정렬
        sorted_data = sorted(self.ranking_data, key=lambda x: x['Accuracy'], reverse=True)

        # 순위를 나타내는 열 추가
        for rank, entry in enumerate(sorted_data, start=1):
            entry['Rank'] = rank

        # DataFrame으로 변환
        ranking_data_df = pd.DataFrame(sorted_data)

        # 순위 열을 첫 번째 열로 이동
        cols = ['Rank'] + [col for col in ranking_data_df if col != 'Rank']
        ranking_data_df = ranking_data_df[cols]

        # CSV 파일로 저장
        if file_path:
            ranking_data_df.to_csv(file_path, index=False)