import cv2
import numpy as np
import pandas as pd

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from data import POSE_CLASSES


class PoseClassifier:
    def __init__(self, n_neighbors, pose_classes):
        self.le = preprocessing.LabelEncoder()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors)

        self.pose_classes = pose_classes
        self.x = []
        self.y = []
        self.img_hashes = []
        self.acc = 0

    def fit(self, test_size=0):
        y = []
        for pose_class in self.pose_classes:
            df = pd.read_csv(f'data/{pose_class}.csv')
            self.img_hashes.extend(list(df['hash']))
            df.drop('hash', axis=1, inplace=True)

            self.x.extend(df.to_numpy().tolist())
            self.y.extend([pose_class] * len(df))

        y = list(self.le.fit_transform(self.y))

        if test_size:
            x_train, x_test, y_train, y_test = train_test_split(
                self.x, y, test_size=test_size)
            self.knn.fit(x_train, y_train)
            self.acc = self.knn.score(x_test, y_test)
        else:
            self.knn.fit(self.x, y)

        self.nbrs.fit(self.x)

    def get_dist_means(self, x):
        x = np.asarray(list(x))
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        distances, indexes = self.nbrs.kneighbors(x)
        dist_means = distances.mean(axis=1)

        if len(dist_means) == 1:
            return dist_means[0]
        return dist_means

    def predict(self, x):
        x = np.asarray(list(x))
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        pred_y = self.knn.predict(x)
        pred_y = self.le.inverse_transform(pred_y)

        if len(pred_y) == 1:
            return pred_y[0]
        return pred_y


if __name__ == '__main__':
    test_pc = PoseClassifier(5, POSE_CLASSES)
    test_pc.fit(test_size=0.2)
    print(f'test_pc acc: {test_pc.acc}')
