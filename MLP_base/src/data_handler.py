import glob
import pickle

class SuperDataSet:
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
        self.split = split
    def load_dataset(self):
        """
        parameters: 
            path: string
            the folder address of the data.
        returns: 
            input_data: dict
            A dict of 4 data splits and 2 price matrix."""    
        input_data = {}
        file_names = [f.replace(self.path,"").replace(".pkl", "") for f in glob.glob(self.path + str(self.split) + "/*.pkl")]
        for file_name in file_names:
            with open(self.path + file_name + '.pkl', 'rb') as f:
                input_data[file_name] = pickle.load(f)
                logging.info(file_name, ' loaded')
                print(file_name, ' loaded')

        return input_data

    def dense_to_sparse(self):
        return
    
    def preprocessor(self):
        return

class ml1m(SuperDataSet):
    def __init__(self, name='ml1m'):
        path = '../../data/ml1m/'
        split=1
        super().__init__(name, path, split)
        
