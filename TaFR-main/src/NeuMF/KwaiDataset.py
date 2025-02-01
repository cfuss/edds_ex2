import numpy as np
from scipy.sparse import coo_matrix


class KwaiDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()

    def load_data(self):
        # Load the dataset
        data = np.loadtxt(self.data_path, delimiter=',', skiprows=1)

        # Assuming the columns in data are: user_id, item_id, survival_prob, base_score, base_rank, base_group, SA_rank, SA_group
        self.survival_prob = data[:, 2]  # Survival probability (used as the rating)

        # Item features
        self.base_score = data[:, 3]
        self.base_rank = data[:, 4]
        self.base_group = data[:, 5]
        self.SA_rank = data[:, 6]
        self.SA_group = data[:, 7]

        # Create the train matrix using user-item pairs and survival probability as ratings
        self.trainMatrix = self.create_train_matrix(data)

        # Create test data with negative sampling
        self.testRatings, self.testNegatives = self.create_test_data(data)

    def create_train_matrix(self, data):
        # Extract user-item pairs
        rows = data[:, 0].astype(int)  # User IDs
        cols = data[:, 1].astype(int)  # Item IDs
        values = self.survival_prob  # Ratings (survival probability)

        # Create a sparse matrix for the training data
        return coo_matrix((values, (rows, cols)))

    def create_test_data(self, data):
        testRatings = []
        testNegatives = []

        # Convert the train matrix to a dense matrix to easily identify user-item interactions
        train_dense = self.trainMatrix.toarray()

        # Loop through users to generate positive and negative samples
        num_users = train_dense.shape[0]
        num_items = train_dense.shape[1]

        for user_id in range(num_users):
            # Positive samples: all items that the user has interacted with
            positive_items = np.where(train_dense[user_id] > 0)[0]

            # Add the positive samples to testRatings
            for item_id in positive_items:
                testRatings.append((user_id, item_id))

            # Negative samples: randomly sample items the user hasn't interacted with
            negative_samples = self.sample_negative_items(user_id, positive_items, num_items)

            # Add negative samples to testNegatives
            for neg_item_id in negative_samples:
                testNegatives.append((user_id, neg_item_id))

        return testRatings, testNegatives

    def sample_negative_items(self, user_id, positive_items, num_items, num_negatives=4):
        # Randomly sample 'num_negatives' items that the user has not interacted with
        negative_samples = []
        while len(negative_samples) < num_negatives:
            # Randomly pick an item that is not in the user's positive items
            neg_item_id = np.random.randint(num_items)
            while neg_item_id in positive_items or neg_item_id in negative_samples:
                neg_item_id = np.random.randint(num_items)
            negative_samples.append(neg_item_id)
        return negative_samples
