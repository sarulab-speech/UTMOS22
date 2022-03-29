
from pathlib import Path
from logging import getLogger

import numpy as np
import pandas as pd

logger = getLogger(__name__)

def make_stage1_data(fold_file, utt_data_dir):

    df = pd.read_csv(fold_file, header=None, index_col=0,
                        names=['wavname', 'true_mos'])
    df = df.sort_index()

    embeddings = []
    for wavname in df.index:
        wavbase = wavname.split('.')[0]
        embeddings.append(np.load(utt_data_dir / f'{wavbase}.npy'))

    X = np.stack(embeddings)
    y = df['true_mos'].values

    return df, X, y

def load_stage1_data(datatrack, ssl_type, i_cv):
    utt_data_dir = Path('../out/utt_data') / ssl_type
    fold_dir = Path('../out/ensemble-multidomain/fold') / datatrack

    logger.info('load data')

    data = {}
    for split in ['train', 'val', 'test']:
        fold_file = fold_dir / f'{split}-{i_cv}.csv'
        df, X, y = make_stage1_data(fold_file, utt_data_dir)

        logger.info('[{}]\tX: {}, y: {}'.format(split, X.shape, y.shape))

        data[split] = {'X': X, 'y': y, 'df': df}

    return data


def load_stage1_train_all_data(datatrack, ssl_type):
    utt_data_dir = Path('../out/utt_data') / ssl_type
    fold_dir = Path('../out/ensemble-multidomain/fold') / datatrack

    logger.info('load data')

    fold_file = fold_dir / f'train-all.csv'
    df, X, y = make_stage1_data(fold_file, utt_data_dir)

    logger.info('[train-all]\tX: {}, y: {}'.format(X.shape, y.shape))

    train_data =  {'X': X, 'y': y, 'df': df}

    return train_data


def load_stage1_test_data(datatrack, ssl_type):
    utt_data_dir = Path('../out/utt_data') / ssl_type
    fold_dir = Path('../out/ensemble-multidomain/fold') / datatrack

    logger.info('load data')

    fold_file = fold_dir / f'test-0.csv'
    df, X, y = make_stage1_data(fold_file, utt_data_dir)

    logger.info('[test]\tX: {}, y: {}'.format(X.shape, y.shape))

    train_data =  {'X': X, 'y': y, 'df': df}

    return train_data


def make_stage2_data(fold_file, df_all_X):

    df_fold = pd.read_csv(fold_file, header=None, index_col=0,
                        names=['wavname', 'true_mos'])
    df_fold = df_fold.sort_index()

    wavnames = df_fold.index.values

    X = df_all_X.loc[wavnames, :].values
    y = df_fold['true_mos'].values

    return df_fold, X, y


def load_stage2_data(datatrack, feat_type, i_cv):
    data_dir = Path('../out/ensemble-multidomain') / 'data-stage2' / datatrack / feat_type
    fold_dir = Path('../out/ensemble-multidomain/fold') / datatrack

    logger.info('load data')


    data = {}
    for split in ['train', 'val', 'test']:
        data_path = data_dir / '{}-X.csv'.format('test' if split == 'test' else 'train')
        df_all_X = pd.read_csv(data_path, index_col=0)
        fold_file = fold_dir / f'{split}-{i_cv}.csv'
        df_fold, X, y = make_stage2_data(fold_file, df_all_X)

        logger.info('[{}]\tX: {}, y: {}'.format(split, X.shape, y.shape))

        data[split] = {'X': X, 'y': y, 'df': df_fold}

    return data


def load_stage2_train_all_data(datatrack, feat_type):
    data_dir = Path('../out/ensemble-multidomain') / 'data-stage2' / datatrack / feat_type
    fold_dir = Path('../out/ensemble-multidomain/fold') / datatrack

    logger.info('load data')

    data_path = data_dir / f'train-X.csv'
    df_all_X = pd.read_csv(data_path, index_col=0)
    fold_file = fold_dir / f'train-all.csv'
    df, X, y = make_stage2_data(fold_file, df_all_X)

    logger.info('[train-all]\tX: {}, y: {}'.format(X.shape, y.shape))

    train_data = {'X': X, 'y': y, 'df': df}

    return train_data


def load_stage2_test_data(datatrack, feat_type):
    data_dir = Path('../out/ensemble-multidomain') / 'data-stage2' / datatrack / feat_type
    fold_dir = Path('../out/ensemble-multidomain/fold') / datatrack

    logger.info('load data')

    data_path = data_dir / f'test-X.csv'
    df_all_X = pd.read_csv(data_path, index_col=0)
    fold_file = fold_dir / f'test-0.csv'
    df, X, y = make_stage2_data(fold_file, df_all_X)

    logger.info('[test]\tX: {}, y: {}'.format(X.shape, y.shape))

    train_data =  {'X': X, 'y': y, 'df': df}

    return train_data

def make_stage3_data(fold_file, df_all_X):
    return make_stage2_data(fold_file, df_all_X)


def load_stage3_data(datatrack, feat_type, i_cv):
    data_dir = Path('../out/ensemble-multidomain') / 'data-stage3' / datatrack / feat_type
    fold_dir = Path('../out/ensemble-multidomain/fold') / datatrack

    logger.info('load data')


    data = {}
    for split in ['train', 'val', 'test']:
        data_path = data_dir / '{}-X.csv'.format('test' if split == 'test' else 'train')
        df_all_X = pd.read_csv(data_path, index_col=0)
        fold_file = fold_dir / f'{split}-{i_cv}.csv'
        df_fold, X, y = make_stage2_data(fold_file, df_all_X)

        logger.info('[{}]\tX: {}, y: {}'.format(split, X.shape, y.shape))

        data[split] = {'X': X, 'y': y, 'df': df_fold}

    return data


def load_stage3_train_all_data(datatrack, feat_type):
    data_dir = Path('../out/ensemble-multidomain') / 'data-stage3' / datatrack / feat_type
    fold_dir = Path('../out/ensemble-multidomain/fold') / datatrack

    logger.info('load data')


    data_path = data_dir / f'train-X.csv'
    df_all_X = pd.read_csv(data_path, index_col=0)
    fold_file = fold_dir / f'train-all.csv'
    df, X, y = make_stage2_data(fold_file, df_all_X)

    logger.info('[train-all]\tX: {}, y: {}'.format(X.shape, y.shape))

    train_data = {'X': X, 'y': y, 'df': df}

    return train_data


def load_stage3_test_data(datatrack, feat_type):
    data_dir = Path('../out/ensemble-multidomain') / 'data-stage3' / datatrack / feat_type
    fold_dir = Path('../out/ensemble-multidomain/fold') / datatrack

    logger.info('load data')

    data_path = data_dir / f'test-X.csv'
    df_all_X = pd.read_csv(data_path, index_col=0)
    fold_file = fold_dir / f'test-0.csv'
    df, X, y = make_stage3_data(fold_file, df_all_X)

    logger.info('[test]\tX: {}, y: {}'.format(X.shape, y.shape))

    train_data =  {'X': X, 'y': y, 'df': df}

    return train_data


def normalize_score(val):
    """
    >>> normalize_score(1)
    -1.0
    >>> normalize_score(3)
    0 
    >>> normalize_score(5)
    1.0
    """
    return (val - 3.0) / 2.0

def inverse_normalize_score(val):
    """
    >>> inverse_normalize_score(-1)
    1.0 
    >>> inverse_normalize_score(0)
    3.0 
    >>> inverse_normalize_score(1)
    5.0
    """
    return (val * 2.0) + 3.0
