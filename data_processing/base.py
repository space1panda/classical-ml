from abc import abstractclassmethod, ABC


class BaseDatasource(ABC):

    @abstractclassmethod
    def __getitem__(self, index):
        pass

    @abstractclassmethod
    def __len__(self):
        pass