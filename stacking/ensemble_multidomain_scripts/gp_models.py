
from pathlib import Path
from logging import getLogger
import joblib
import os
import json
import pickle

import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error, pairwise_distances
import torch
import torch.utils
import gpytorch

from data_util import normalize_score, inverse_normalize_score

logger = getLogger(__name__)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dx, dy, transform=None):

        self._N = len(dx)
        self._dx = dx
        self._dy = dy
        
        self.transform = transform
    
    def __len__(self):
        return self._N
    
    def __getitem__(self, idx):
        return self._dx[idx], self._dy[idx]

class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, initial_inducing, initial_lengthscale):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(initial_inducing.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, initial_inducing, variational_distribution, learn_inducing_locations=True
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
                                gpytorch.kernels.RBFKernel(ard_num_dims=initial_inducing.size(1)))
        self.covar_module.base_kernel.lengthscale = initial_lengthscale


        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class SVGP:

    def __init__(self, params=None, stage='stage1'):

        if params is None:
            self.params = {
                'max_inducings': 1024,
                'batch_size': 1024,
                # 'training_epochs': 3000 if stage == 'stage1' else 1000,
                'training_iters': 10000 if stage == 'stage1' else 2500,
            }

        else:
            self.params = params

        self.device = torch.device('cuda')

    def get_num_inducing(self, num_data):
        max_inducings = self.params['max_inducings']

        if num_data >= max_inducings:
            return max_inducings

        power = math.floor((math.log(num_data) / math.log(2)))
        num_inducings = int(2 ** power)

        return num_inducings


    def train(self, train_X, train_y, val_X, val_y):

        X_scaler = StandardScaler()
        train_X_sc = torch.from_numpy(X_scaler.fit_transform(train_X).astype(np.float32))
        train_y_sc = torch.from_numpy(normalize_score(train_y).astype(np.float32))

        val_X_sc = torch.from_numpy(X_scaler.transform(val_X).astype(np.float32))

        # dataloader
        train_X_sc = train_X_sc.to(self.device)
        train_y_sc = train_y_sc.to(self.device)
        val_X_sc = val_X_sc.to(self.device)

        dataset = Dataset(train_X_sc, train_y_sc)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)

        # initial inducing
        num_inducings = self.get_num_inducing(len(train_X_sc))
        logger.info('Num inducing points: {}'.format(num_inducings))

        kmeans = MiniBatchKMeans(num_inducings)

        for i in range(5):
            for b, (x_B, y_B) in enumerate(dataloader):
                kmeans.partial_fit(x_B.cpu().numpy())

        initial_inducing = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32))

        # initial lengthscale
        for b, (x_B, y_B) in enumerate(dataloader):
            D = pairwise_distances(x_B.cpu().numpy())

            distances = D[np.tril_indices(len(D), k=-1)]
            # initial_lengthscale = np.sqrt(np.median(distances))
            initial_lengthscale = np.median(distances)

            break

        # initialize likelihood and model
        gpr = SVGPModel(initial_inducing=initial_inducing,
                            initial_lengthscale=initial_lengthscale)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        gpr = gpr.to(self.device)
        likelihood = likelihood.to(self.device)

        # optimizer
        variational_ngd_optimizer = gpytorch.optim.NGD(gpr.variational_parameters(),
                                                        num_data=train_X_sc.size(0), lr=0.1)

        hyperparameter_optimizer = torch.optim.Adam([
            {'params': gpr.hyperparameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)

        # training
        gpr.train()
        likelihood.train()

        mll = gpytorch.mlls.VariationalELBO(likelihood, gpr, num_data=train_X_sc.size(0))

        num_epochs = max(1, self.params['training_iters'] // len(dataloader))
        for i in range(num_epochs):
            gpr.train()
            likelihood.train()

            lower_bound = 0
            for b, (x_B, y_B) in enumerate(dataloader):
                # x_B = x_B.to(device)
                # y_B = y_B.to(device)
                
                variational_ngd_optimizer.zero_grad()
                hyperparameter_optimizer.zero_grad()

                output = gpr(x_B)
                bound = mll(output, y_B)
                loss = -bound
                loss.backward()

                variational_ngd_optimizer.step()
                hyperparameter_optimizer.step()

                lower_bound += bound.data.cpu().item()

            if i % 100 == 0:
                gpr.eval()
                likelihood.eval()
                with torch.no_grad():
                    preds = gpr(val_X_sc)
                    mean = preds.mean
                    pred_y_sc = mean.cpu().numpy()

                    pred_y = inverse_normalize_score(pred_y_sc)

                    mse = mean_squared_error(val_y, pred_y.ravel())

                logger.info('Iter %d/%d - ELBO: %.3f - val_loss %.3f  ' % (
                    i + 1, num_epochs, lower_bound, mse,
                ))

        self.X_scaler = X_scaler
        self.gpr = gpr

        self.conf = {
            'num_inducings': num_inducings,
            'input_dim': train_X.shape[1],
        }

    def predict(self, X, df):
        self.gpr.eval()

        X_sc = torch.from_numpy(self.X_scaler.transform(X).astype(np.float32))
        X_sc = X_sc.to(self.device)

        with torch.no_grad():
            preds = self.gpr(X_sc)
            mean = preds.mean
            lower, upper = preds.confidence_region()

        mean_y = inverse_normalize_score(mean.cpu().numpy())
        lower_y = inverse_normalize_score(lower.cpu().numpy())
        upper_y = inverse_normalize_score(upper.cpu().numpy())

        df['pred_mos'] = mean_y.ravel()
        df['lower_mos'] = lower_y.ravel()
        df['upper_mos'] = upper_y.ravel()

        return df


    def save_model(self, out_dir: Path):
        torch.save(self.gpr.state_dict(), out_dir / 'model.pt')
        joblib.dump(self.X_scaler, out_dir / 'X_scaler.joblib')

        with open(out_dir / 'model_config.json', encoding="utf-8", mode="w") as f:
            json.dump(self.conf, f, ensure_ascii=False, indent=2)


    def load_model(self, model_dir: Path, train_X=None):
        if os.path.exists(model_dir / 'model_config.json'):
            self.conf = json.load(open(model_dir / 'model_config.json', 'rb'))
        else:
            assert train_X is not None
            self.conf = {
                'num_inducings': self.get_num_inducing(len(train_X)),
                'input_dim': train_X.shape[1],
            }

        initial_inducing = torch.from_numpy(
            np.empty((self.conf['num_inducings'], self.conf['input_dim']), dtype=np.float32))

        self.gpr = SVGPModel(initial_inducing, 1.0)
        self.gpr.to(self.device)

        self.gpr.load_state_dict(torch.load(model_dir / 'model.pt', map_location=self.device))
        self.X_scaler = joblib.load(model_dir / 'X_scaler.joblib')

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, initial_lengthscale):    
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        # lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
        # outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                # lengthscale_prior=lengthscale_prior,
                ard_num_dims=train_x.size(1),
            ),
            # outputscale_prior=outputscale_prior
        )

        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(
        #         lengthscale_prior=lengthscale_prior,
        #         ard_num_dims=train_x.size(1),
        #     ),
        #     outputscale_prior=outputscale_prior
        # )


        # Initialize lengthscale and outputscale to mean of priors
        # self.covar_module.base_kernel.lengthscale = initial_lengthscale
        self.covar_module.base_kernel.lengthscale = initial_lengthscale
        # self.covar_module.outputscale = outputscale_prior.mean



    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGP:

    def __init__(self, params=None, stage='stage1'):
        if params is None:
            self.params = {
                # 'max_inducings': 1024,
                # 'batch_size': 1024,
                # 'training_epochs': 3000 if stage == 'stage1' else 1000,
                # 'training_iters': 10000 if stage == 'stage1' else 2500,
                'training_iters': 10000,
            }

        else:
            self.params = params

        self.device = torch.device('cuda')

    def train(self, train_X, train_y, val_X, val_y):

        X_scaler = StandardScaler()
        train_X_sc = torch.from_numpy(X_scaler.fit_transform(train_X).astype(np.float32))
        train_y_sc = torch.from_numpy(normalize_score(train_y).astype(np.float32))

        val_X_sc = torch.from_numpy(X_scaler.transform(val_X).astype(np.float32))

        # dataloader
        train_X_sc = train_X_sc.to(self.device)
        train_y_sc = train_y_sc.to(self.device)
        val_X_sc = val_X_sc.to(self.device)

        # dataset = Dataset(train_X_sc, train_y_sc)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True)

        # initial lengthscale
        D = pairwise_distances(train_X_sc.cpu().numpy())

        distances = D[np.tril_indices(len(D), k=-1)]
        # initial_lengthscale = np.sqrt(np.median(distances))
        initial_lengthscale = np.median(distances)

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
            )
        gpr = ExactGPModel(train_X_sc, train_y_sc, likelihood,
                            initial_lengthscale=initial_lengthscale)

        gpr = gpr.to(self.device)
        likelihood = likelihood.to(self.device)

        # optimizer
        optimizer = torch.optim.Adam(gpr.parameters(), lr=0.01)

        # training
        gpr.train()
        likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr)

        for i in range(self.params['training_iters']):
            gpr.train()
            likelihood.train()

            optimizer.zero_grad()

            output = gpr(train_X_sc)
            loss = -mll(output, train_y_sc)
            loss.backward()

            optimizer.step()

            if i % 100 == 0:
                gpr.eval()
                likelihood.eval()
                with torch.no_grad():
                    preds = gpr(val_X_sc)
                    mean = preds.mean
                    pred_y_sc = mean.cpu().numpy()

                    pred_y = inverse_normalize_score(pred_y_sc)

                    mse = mean_squared_error(val_y, pred_y.ravel())

                logger.info('Iter %d/%d - NMLL: %.3f - val_loss %.3f  ' % (
                    i + 1, self.params['training_iters'], loss, mse,
                ))
                # logger.info('Pred_y var %.3f  ' % (
                #     pred_y.var(),
                # ))

        self.X_scaler = X_scaler
        self.gpr = gpr

        self.conf = {
            'output_shape': train_y.shape,
            'input_shape': train_X.shape,
        }

    def predict(self, X, df):
        self.gpr.eval()

        X_sc = torch.from_numpy(self.X_scaler.transform(X).astype(np.float32))
        X_sc = X_sc.to(self.device)

        with torch.no_grad():
            preds = self.gpr(X_sc)
            mean = preds.mean
            lower, upper = preds.confidence_region()

        mean_y = inverse_normalize_score(mean.cpu().numpy())
        lower_y = inverse_normalize_score(lower.cpu().numpy())
        upper_y = inverse_normalize_score(upper.cpu().numpy())

        df['pred_mos'] = mean_y.ravel()
        df['lower_mos'] = lower_y.ravel()
        df['upper_mos'] = upper_y.ravel()

        return df


    def save_model(self, out_dir: Path):
        torch.save(self.gpr.state_dict(), out_dir / 'model.pt')
        joblib.dump(self.X_scaler, out_dir / 'X_scaler.joblib')
        # pickle.dump(self.gpr, open(out_dir / 'gpr_pkl.pkl', 'wb'))

        with open(out_dir / 'model_config.json', encoding="utf-8", mode="w") as f:
            json.dump(self.conf, f, ensure_ascii=False, indent=2)


    def load_model(self, model_dir: Path, train_X, train_y):
        self.conf = json.load(open(model_dir / 'model_config.json', 'rb'))

        self.X_scaler = joblib.load(model_dir / 'X_scaler.joblib')

        train_X_sc = torch.from_numpy(self.X_scaler.transform(train_X).astype(np.float32))
        train_y_sc = torch.from_numpy(normalize_score(train_y).astype(np.float32))

        # dummy_X = torch.from_numpy(
        #     np.empty(self.conf['input_shape'], dtype=np.float32))
        # dummy_y = torch.from_numpy(
        #     np.empty(self.conf['output_shape'], dtype=np.float32))

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
                        noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
                    )
        self.gpr = ExactGPModel(train_X_sc, train_y_sc, likelihood,
                            initial_lengthscale=1.0)
        self.gpr.to(self.device)

        self.gpr.load_state_dict(torch.load(model_dir / 'model.pt', map_location=self.device))

