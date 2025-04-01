import random
from sklearn.model_selection import train_test_split
from pathlib import Path
import pyterrier as pt

class DatasetSetup:
    def __init__(self, name, index, train, test, primary_field="abstract", metadata=["docno", "title", "abstract", "url"], dev=None, split=0.2, rand=random.randint(0, 1000)):
        self.name = name
        self.index = index
        self.test = test
        self.train = train
        self.dev = dev
        self.split = split
        self.rand = rand
        self.primary_field = primary_field
        self.metadata = metadata

    def get_train(self):
        if self.dev is None:
            topics_train, topics_val = train_test_split(
                self.train.get_topics(), test_size=self.split, random_state=self.rand
            )
            queries_train, queries_val = train_test_split(
                self.train.get_qrels(), test_size=self.split, random_state=self.rand
            )

            return topics_train, queries_train, topics_val, queries_val
        else:
            return self.train.get_topics(), self.train.get_qrels(), self.dev.get_topics(), self.dev.get_qrels()
    
    def get_test(self):
        return self.test.get_topics(), self.test.get_qrels()
    

def NFCorpus():
    dataset = pt.get_dataset("irds:nfcorpus")

    index = pt.index.IterDictIndexer(
        str(Path.cwd()),  # this will be ignored
        meta={
            "docno": 16,
            "title": 256,
            "abstract": 65536,
            "url": 128,
        },
        type=pt.index.IndexingType.MEMORY,
    ).index(dataset.get_corpus_iter(), fields=["title", "abstract", "url"])

    return DatasetSetup(
        name="NF_Corpus",
        index=index,
        train=pt.get_dataset("irds:nfcorpus/train/nontopic"),
        test=pt.get_dataset("irds:nfcorpus/test/nontopic"),
        dev=pt.get_dataset("irds:nfcorpus/dev/nontopic"),
    )

def Antique():
    dataset = pt.datasets.get_dataset("irds:antique")
    index = pt.index.IterDictIndexer(
        str(Path.cwd()),  # this will be ignored
        meta={
            "docno": 16,
            "text": 131072,
        },
        type=pt.index.IndexingType.MEMORY,
    ).index(dataset.get_corpus_iter(), fields=["text"])

    return DatasetSetup(
        name="Antique",
        index=index,
        train=pt.get_dataset("irds:antique/train"),
        test=pt.get_dataset("irds:antique/test"),
        primary_field="text",
        metadata=["docno", "text"],
    )