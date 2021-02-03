import glob
import pickle
import logging
a_logger = logging.getLogger()
a_logger.setLevel(logging.DEBUG)

class DataSet:
    """ A class of datasets, that loads, modifies and process their data."""
    def __init__(self, name, path, split):
        """
        Parameters:
            name: str
                The name of the dataset
            path: str
                The absolute path to the main dataset folder where all its splits are
            split: int
                The nth split of the dataset in case of cross validatoin.  """
        
        self.name = name
        self.path = path
        self.split = str(split)
    def load_dataset(self):
        """
        parameters: 
            path: string
            the folder address of the data.
        returns: 
            input_data: dict
            A dict of 4 data splits and 2 price matrix."""    
        input_data = {}
        file_names = \
            [f.replace(self.path + self.split + "/", "").
            replace(".pkl", "") for f in
            glob.glob(self.path + self.split + "/*.pkl")]

        for file_name in file_names:
            with open(self.path + self.split + "/" + file_name + '.pkl', 'rb') as f:
                input_data[file_name] = pickle.load(f)
                print("{0} loaded".format(file_name))

        return input_data

    def dense_to_sparse(self):
        return
    
    def preprocessor(self):
        return

class Ml1m(DataSet):
    def __init__(self, name='ml1m'):
        path = '../../data/ml1m/'
        split=1
        super().__init__(name, path, split)
        
