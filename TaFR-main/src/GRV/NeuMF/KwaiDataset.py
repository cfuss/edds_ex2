import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


class KwaiDataset:
    def __init__(self, data_path, num_negatives=4):
        self.data_path = data_path
        self.num_negatives = num_negatives
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.data_path)

        df = df.iloc[:10, :]

        date_columns = [col for col in df.columns if col.isdigit()]
        df['SA_score'] = df[date_columns].sum(axis=1)

        self.user_ids = df['photo_id'].values
        self.item_ids = np.arange(len(df))
        self.clicks = df['click'].values
        self.counters = df['counter'].values
        self.play_rate = df['play_rate'].values
        self.click_rate = df['click_rate'].values
        self.ctr = df['ctr'].values
        self.base_score = df['base_score'].values
        self.base_rank = df['base_rank'].values
        self.base_group = df['base_group'].values
        self.SA_rank = df['SA_rank'].values
        self.SA_group = df['SA_group'].values
        self.trainMatrix = self.create_train_matrix(df)
        self.testRatings, self.testNegatives = self.create_test_data()

    def create_train_matrix(self, df):
        """
        Creates a sparse training matrix where:
        - Rows correspond to users
        - Columns correspond to items
        - Values are the interaction strength (e.g., click, play_rate, or survival probability)
        """
        rows = self.user_ids
        cols = self.item_ids
        values = df['SA_score'].values  # Using SA_score as the implicit feedback strength

        return coo_matrix((values, (rows, cols)), shape=(max(rows) + 1, len(cols)))

    def create_test_data(self):
        """
        Creates test data by selecting positive interactions and sampling negative examples.
        """
        testRatings = []
        testNegatives = []

        # Convert the train matrix to a dense format for easier lookups
        train_dense = self.trainMatrix.toarray()

        num_users, num_items = train_dense.shape

        for user_id in range(num_users):
            # Positive items: Items interacted with by the user
            positive_items = np.where(train_dense[user_id] > 0)[0]

            # Adding positive samples to testRatings
            for item_id in positive_items:
                testRatings.append((user_id, item_id))

            # Negative sampling: Pick random items not interacted with
            negative_samples = self.sample_negative_items(user_id, positive_items, num_items)
            for neg_item_id in negative_samples:
                testNegatives.append((user_id, neg_item_id))

        return testRatings, testNegatives

    def sample_negative_items(self, user_id, positive_items, num_items):
        """
        Randomly samples negative items for a given user.
        """
        negative_samples = []
        while len(negative_samples) < self.num_negatives:
            neg_item_id = np.random.randint(num_items)
            if neg_item_id not in positive_items and neg_item_id not in negative_samples:
                negative_samples.append(neg_item_id)
        return negative_samples