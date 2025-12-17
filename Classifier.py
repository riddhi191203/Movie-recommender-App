import numpy as np
from operator import itemgetter


class KNearestNeighbours:
    def __init__(self, data, target, test_point, k):
        self.data = data
        self.target = target
        self.test_point = test_point
        self.k = k
        self.distances = []
        self.categories = []
        self.indices = []
        self.counts = []
        self.category_assigned = None

    @staticmethod
    def dist(p1, p2):
        """Returns Euclidean distance"""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def fit(self):
        """Perform KNN"""

        # 🔥 RESET STATE (VERY IMPORTANT)
        self.distances = []
        self.indices = []
        self.categories = []
        self.counts = []

        # Calculate distances
        for i, point in enumerate(self.data):
            d = self.dist(self.test_point, point)
            self.distances.append((d, i))

        # Sort distances
        sorted_li = sorted(self.distances, key=itemgetter(0))

        # Get k nearest indices
        self.indices = [index for (_, index) in sorted_li[:self.k]]

        # Get categories
        for i in self.indices:
            self.categories.append(self.target[i])

        # Count categories
        self.counts = [(i, self.categories.count(i)) for i in set(self.categories)]

        # Assign category
        self.category_assigned = sorted(
            self.counts,
            key=itemgetter(1),
            reverse=True
        )[0][0]
