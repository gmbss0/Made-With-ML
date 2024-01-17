from typing import Dict, List, Tuple 
import re

import ray
from ray.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords as swords

from transformers import Tokenizer, BertTokenizer


def load_dataset(path: str, n: int = None) -> Dataset:
    """ Load data into Ray Dataset.

    Args:
        path (str): path to data.
        n (int): amount of samples to load. Defaults to None.
    
    Returns:
        Dataset: Ray Dataset
    """
    ds = ray.data.read_csv(path)
    ds = ds.random_shuffle(seed=10)  # shuffle data
    ds = ray.data.from_items(ds.take(n)) if n else ds

    return ds

def split_stratify(ds: Dataset, col: str,
                   split: float, shuffle: bool = True,
                   seed: int = 10) -> Tuple[Dataset, Dataset]:
    """Split dataset into equal samples from each class in the column we
    want to stratify on.

    Args:
        ds (Dataset): Input dataset.
        col (str): Name of column to stratify on.
        split (float): Proportion of dataset to split for test set.
        shuffle (bool, optional): whether to shuffle the dataset. Defaults to True.
        seed (int, optional): seed for shuffling. Defaults to 10.

    Returns:
        Tuple[Dataset, Dataset]: the stratified train and test datasets.
    """
    def _add_split(df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover, used in parent function
        """Naively split a dataframe into train and test splits.
        Add a column specifying whether it's the train or test split."""
        train, test = train_test_split(df, test_size=split, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:  # pragma: no cover, used in parent function
        """Filter by samples that match the split column's value
        and return the dataframe with the _split column dropped."""
        return df[df["_split"] == split].drop("_split", axis=1)

    # stratified split
    grouped = ds.groupby(col).map_groups(_add_split, batch_format="pandas")  # group by each unique value in the column we want to stratify on
    train_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "train"}, batch_format="pandas") 
    test_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "test"}, batch_format="pandas") 

    # shuffle
    if shuffle:
        train_ds = train_ds.random_shuffle(seed=seed)
        test_ds = test_ds.random_shuffle(seed=seed)

    return train_ds, test_ds


def clean_text(text: str, stopwords: List = None) -> str:
    """Clean raw text.

    Args:
        text (str): Raw text to clean.
        stopwords (List, optional): list of words to filter out. Defaults to None -> nltk.corpus.stopwords are used.

    Returns:
        str: cleaned text.
    """
    if not stopwords:
        stopwords = swords.words("english")
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  # remove links

    return text


def tokenize(batch: Dict, tokenizer: Tokenizer = BertTokenizer) -> Dict:
    """Tokenize the text input in our batch using a tokenizer.

    Args:
        batch (Dict): batch of data with the text inputs to tokenize.

    Returns:
        Dict: batch of tokenized input text.
    """
    if isinstance(tokenizer, BertTokenizer):
        tokenizer = tokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return dict(ids=encoded_inputs["input_ids"], masks=encoded_inputs["attention_mask"], targets=np.array(batch["tag"]))


def preprocess(df: pd.DataFrame, class_to_index: Dict) -> Dict:
    """Preprocess the data.

    Args:
        df (pd.DataFrame): Raw dataframe to preprocess.
        class_to_index (Dict): Mapping of class names to indices.

    Returns:
        Dict: preprocessed data (ids, masks, targets).
    """
    df["text"] = df.title + " " + df.description  # combine title and description
    df["text"] = df.text.apply(clean_text)  # clean text
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")
    df = df[["text", "tag"]]  # rearrange columns
    df["tag"] = df["tag"].map(class_to_index)  # label encoding
    outputs = tokenize(df)
    return outputs


class CustomPreprocessor:
    """Custom preprocessor class to ."""

    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  # mutable defaults
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

    def fit(self, ds):
        tags = ds.unique(column="tag")
        self.class_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        return self
    
    def transform(self, ds):
        return ds.map_batches(preprocess, fn_kwargs={"class_to_index": self.class_to_index}, batch_format="pandas")
