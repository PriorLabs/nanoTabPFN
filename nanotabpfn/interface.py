import os

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from numpy import ndarray
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.utils import get_default_device


def init_model_from_state_dict_file(file_path):
    """
    infers model architecture from state dict, instantiates the architecture and loads the weights
    """
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    embedding_size = state_dict['feature_encoder.linear_layer.weight'].shape[0]
    mlp_hidden_size = state_dict['decoder.linear1.weight'].shape[0]
    num_outputs = state_dict['decoder.linear2.weight'].shape[0]
    num_layers = sum('self_attn_between_datapoints.q_proj.weight' in k for k in state_dict)
    num_heads = state_dict['transformer_encoder.transformer_blocks.0.self_attn_between_datapoints.q_proj.weight'].shape[1]//64
    model = NanoTabPFNModel(
        num_attention_heads=num_heads,
        embedding_size=embedding_size,
        mlp_hidden_size=mlp_hidden_size,
        num_layers=num_layers,
        num_outputs=num_outputs,
    )
    model.load_state_dict(torch.load(file_path, map_location='cpu'))
    return model

def get_feature_preprocessor(X: ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs
    """
    X = pd.DataFrame(X)
    num_mask = []
    for col in X:
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum() # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x)), # to apply pd.to_numeric of pandas
        ("to_numeric", FunctionTransformer(lambda x: x.apply(pd.to_numeric, errors='coerce').to_numpy())), # in case numeric columns are stored as strings
        ('imputer', SimpleImputer(strategy='mean')) # median might be better because of outliers
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, ~num_mask)
        ]
    )
    return preprocessor


class NanoTabPFNClassifier():
    """ scikit-learn like interface """
    def __init__(self, model: NanoTabPFNModel|str|None = None, device=get_default_device()):
        if model == None:
            model = 'nanotabpfn.pth'
            if not os.path.isfile(model):
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/nanotabpfn_classifier.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        self.model = model.to(device)
        self.device = device

    def fit(self, X_train: np.array, y_train: np.array):
        """ stores X_train and y_train for later use, also computes the highest class number occuring in num_classes """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = max(set(y_train))+1

    def predict(self, X_test: np.array) -> np.array:
        """ calls predit_proba and picks the class with the highest probability for each datapoint """
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.array) -> np.array:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y), single_eval_pos=len(self.X_train)).squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, :self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to('cpu').numpy()


class NanoTabPFNRegressor():
    """ scikit-learn like interface """
    def __init__(self, model: NanoTabPFNModel|str|None = None, dist: FullSupportBarDistribution|str|None = None, device=get_default_device()):
        if model is None:
            model = 'nanotabpfn_regressor.pth'
            dist = 'nanotabpfn_regressor_buckets.pth'
            if not os.path.isfile(model):
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/nanotabpfn_regressor.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print('No cached bucket edges found, downloading bucket edges.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/nanoTabPFN/nanotabpfn_regressor_buckets.pth')
                with open(dist, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)

        if isinstance(dist, str):
            bucket_edges = torch.load(dist, map_location=device)
            dist = FullSupportBarDistribution(bucket_edges).float()

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.normalized_dist = None  # Used after fit()

    def fit(self, X_train: np.array, y_train: np.array):
        """
        Stores X_train and y_train for later use. Computes target normalization. Builds normalized bar distribution from existing self.dist.
        """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train) + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

        # Convert base distribution to original scale for output
        bucket_edges = self.dist.borders
        bucket_edges_denorm = bucket_edges * self.y_train_std + self.y_train_mean
        self.normalized_dist = FullSupportBarDistribution(bucket_edges_denorm).float()

    def predict(self, X_test: np.array) -> np.array:
        """
        Performs in-context learning using X_train and y_train. Predicts the means of the output distributions for X_test.
        """
        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

            logits = self.model((X_tensor, y_tensor), single_eval_pos=len(self.X_train)).squeeze(0)
            preds = self.normalized_dist.mean(logits)

        return preds.cpu().numpy()
