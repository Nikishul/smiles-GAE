import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


class SmilesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self, max_length=120, exclude_atoms=["p", "e", "%"], exclude_cycles=list(np.arange(10, 100, 1).astype(str))
    ):

        self.max_length = max_length
        self.exclude_atoms = exclude_atoms
        self.exclude_cycles = exclude_cycles

    def fit(self, X):

        self.charset = set("".join(list(X["smiles"])) + "!E")
        self.char_to_int = dict((c, i) for i, c in enumerate(self.charset))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.charset))
        self.embed = max([len(smile) for smile in X["smiles"]]) + 5

        return self

    def transform(self, X):

        X["len"] = X["smiles"].apply(len)
        df = X[X["len"] <= self.max_length]

        df = df[df["smiles"].apply(lambda x: not any(char in x for char in self.exclude_atoms))]
        df = df[df["smiles"].apply(lambda x: not any(x.count(symbol) > 1 for symbol in self.exclude_cycles))]

        df = df.drop_duplicates(subset=["smiles"], keep="first")

        smiles_train, smiles_test, energy_train, energy_test = self.prepare_train_test(df)
        X_train, Y_train = self.vectorize(smiles_train)
        X_test, Y_test = self.vectorize(smiles_test)

        return X_train, Y_train, X_test, Y_test, energy_train, energy_test

    def vectorize(self, smiles):
        one_hot = np.zeros((smiles.shape[0], self.embed, len(self.charset)), dtype=np.int8)
        for i, smile in enumerate(smiles):
            one_hot[i, 0, self.char_to_int["!"]] = 1

            for j, c in enumerate(smile):
                one_hot[i, j + 1, self.char_to_int[c]] = 1

            one_hot[i, len(smile) + 1 :, self.char_to_int["E"]] = 1

        return one_hot[:, 0:-1, :], one_hot[:, 1:, :]

    @staticmethod
    def prepare_train_test(df, min_energy=-12, max_energy=-6, n_bins=13, test_size=0.15, random_state=12):
        bins = np.linspace(min_energy, max_energy, n_bins)
        y_binned = np.digitize(df["vina"], bins)
        smiles_train, smiles_test, energy_train, energy_test = train_test_split(
            df["smiles"], df["vina"], stratify=y_binned, test_size=test_size, random_state=random_state
        )

        return smiles_train, smiles_test, energy_train, energy_test
