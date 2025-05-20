from sklearn.datasets import fetch_20newsgroups

class NG20:
    def __init__(self, remove_metadata=True, categories=None):
        """
        Initialize NG20 loader.
        
        Args:
            remove_metadata (bool): If True, removes headers, footers, and quotes.
            categories (list or None): List of newsgroup categories to load.
        """
        self.remove = ('headers', 'footers', 'quotes') if remove_metadata else ()
        self.categories = categories
        self.train_data = None
        self.test_data = None

    def load_train(self):
        """Load the training subset of 20 Newsgroups."""
        self.train_data = fetch_20newsgroups(
            subset='train',
            remove=self.remove,
            categories=self.categories
        )
        return self.train_data

    def load_test(self):
        """Load the test subset of 20 Newsgroups."""
        self.test_data = fetch_20newsgroups(
            subset='test',
            remove=self.remove,
            categories=self.categories
        )
        return self.test_data

    def get_sample(self, subset='train', index=0):
        """Get a sample text and label from the specified subset.

        Args:
            subset (str): 'train' or 'test'
            index (int): Index of the sample

        Returns:
            tuple: (text, label_index, label_name)
        """
        data = self.train_data if subset == 'train' else self.test_data
        if data is None:
            raise ValueError(f"{subset} data not loaded. Call load_{subset}() first.")
        text = data.data[index]
        label_index = data.target[index]
        label_name = data.target_names[label_index]
        return text, label_index, label_name

    def get_all_texts(self):
        self.load_train()
        self.load_test()
        texts = self.train_data.data + self.test_data.data
        return texts