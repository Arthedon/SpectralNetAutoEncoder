from Model.SpectralNetAEModel import SpectralNetAEModel
from Loss.SpectralNetAELoss import SpectralNetAELoss
from spectralnet._utils import make_batch_for_sparse_grapsh
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.neighbors import kneighbors_graph
from tqdm.auto import tqdm
from tqdm import trange


class SpectralNetAETrainer:
    def __init__(self, config: dict, device: torch.device, is_sparse: bool = False):
        """
        Initialize the SpectralNetAE model trainer.

        Parameters
        ----------
        config : dict
            The configuration dictionary.
        device : torch.device
            The device to use for training.
        is_sparse : bool, optional
            Whether the graph-laplacian obtained from a mini-batch is sparse or not.
            If True, the batch is constructed by taking 1/5 of the original random batch
            and adding 4 of its nearest neighbors to each sample. Defaults to False.

        Notes
        -----
        This class is responsible for training the SpectralNet model.
        The configuration dictionary (`config`) contains various settings for training.
        The device (`device`) specifies the device (CPU or GPU) to be used for training.
        The `is_sparse` flag is used to determine the construction of the batch when the graph-laplacian is sparse.
        """

        self.device = device
        self.is_sparse = is_sparse
        self.spectral_config = config
        self.lr = self.spectral_config["lr"]
        self.num_neighbours = self.spectral_config["num_neighbours"]
        self.min_lr = self.spectral_config["min_lr"]
        self.max_epochs = self.spectral_config["max_epochs"]
        self.tolerance = self.spectral_config["tolerance"]
        self.scale_k = self.spectral_config["scale_k"]
        self.lr_decay = self.spectral_config["lr_decay"]
        self.patience = self.spectral_config["patience"]
        self.architecture = self.spectral_config["architecture"]
        self.batch_size = self.spectral_config["batch_size"]
        self.is_local_scale = self.spectral_config["is_local_scale"]
        self.is_normalized = self.spectral_config["is_normalized"]
        self.alpha = self.spectral_config["alpha"]
        self.use_convolution = self.spectral_config["use_convolution"]

    def train(
        self, X: torch.Tensor, y: torch.Tensor, siamese_net: nn.Module = None
    ) -> SpectralNetAEModel:
        """
        Train the SpectralNet model.

        Parameters
        ----------
        X : torch.Tensor
            The dataset to train on.
        y : torch.Tensor, optional
            The labels of the dataset in case there are any.
        siamese_net : nn.Module, optional
            The siamese network to use for computing the affinity matrix.

        Returns
        -------
        SpectralNetModel
            The trained SpectralNet model.

        Notes
        -----
        This function trains the SpectralNet model using the provided dataset (`X`) and labels (`y`).
        If labels are not provided (`y` is None), unsupervised training is performed.
        The siamese network (`siamese_net`) is an optional parameter used for computing the affinity matrix.
        The trained SpectralNet model is returned as the output.
        """
        if self.use_convolution == False:
            self.X = X.view(X.size(0), -1)
        else:
            self.X = X
        self.y = y
        self.counter = 0
        self.siamese_net = siamese_net
        self.criterion = SpectralNetAELoss()

        affinity_config = {
            'scale_k': self.scale_k,
            'num_neighbours': self.num_neighbours,
            'is_local_scale': self.is_local_scale,
            'device': self.device,
        }

        self.spectral_net_ae = SpectralNetAEModel(
            affinity_config, self.architecture, input_dim = self.X.shape[1], use_convolution = self.use_convolution
        ).to(self.device)

        self.optimizer = optim.Adam(self.spectral_net_ae.parameters(), lr = self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode = "min", factor = self.lr_decay, patience = self.patience
        )

        train_loader, ortho_loader, valid_loader = self._get_data_loader()

        print("Training SpectralNetAE:")
        progress_bar = tqdm(total = self.max_epochs, leave = True)

        # Keep track of the loss and the SpectralNet loss during the epochs
        spectral_train_loss_count, spectral_valid_loss_count = [], []
        train_loss_count, valid_loss_count = [], []

        prec_valid_loss = 0
        valid_loss = - 2 * self.tolerance
        epoch = 0

        while ((np.abs(valid_loss - prec_valid_loss) > self.tolerance) and (epoch < self.max_epochs)):
            train_loss, spectral_train_loss = 0.0, 0.0
            epoch += 1

            for (X_grad, _), (X_orth, _) in zip(train_loader, ortho_loader):
                X_grad = X_grad.to(device = self.device)
                if not self.use_convolution:
                    X_grad = X_grad.view(X_grad.size(0), -1)
                X_orth = X_orth.to(device = self.device)
                if not self.use_convolution:
                    X_orth = X_orth.view(X_orth.size(0), -1)

                if self.is_sparse:
                    X_grad = make_batch_for_sparse_grapsh(X_grad)
                    X_orth = make_batch_for_sparse_grapsh(X_orth)

                # Orthogonalization step
                self.spectral_net_ae.eval()
                self.spectral_net_ae.encode(X_orth, update_orth_weights = True)

                # Gradient step
                self.spectral_net_ae.train()
                self.optimizer.zero_grad()

                X_grad_out = self.spectral_net_ae(X_grad)

                # Computed during the forward propagation
                spectral_train_loss += self.spectral_net_ae.SpectralNetLoss

                loss = self.criterion(X_grad, X_grad_out, self.spectral_net_ae.SpectralNetLoss, self.alpha)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            spectral_train_loss /= len(train_loader)

            spectral_train_loss_count.append(spectral_train_loss.detach().cpu())
            train_loss_count.append(train_loss)

            prec_valid_loss = valid_loss
            
            # Validation step
            valid_loss, spectral_valid_loss = self.validate(valid_loader)

            spectral_valid_loss_count.append(spectral_valid_loss.detach().cpu())
            valid_loss_count.append(valid_loss)

            self.scheduler.step(valid_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]:
                break
            progress_bar.n = epoch
            progress_bar.set_description(
                "Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(
                    train_loss, valid_loss, current_lr
                )
            )
            progress_bar.refresh()

        return self.spectral_net_ae, train_loss_count, valid_loss_count, spectral_train_loss_count, spectral_valid_loss_count

    def validate(self, valid_loader: DataLoader) -> float:
        valid_loss, spectral_valid_loss = 0.0, 0.0
        self.spectral_net_ae.eval()
        with torch.no_grad():
            for batch in valid_loader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)

                if self.is_sparse:
                    X = make_batch_for_sparse_grapsh(X)

                X_out = self.spectral_net_ae(X, self.is_normalized, self.siamese_net)

                loss = self.criterion(X, X_out, self.spectral_net_ae.SpectralNetLoss, self.alpha)
                spectral_valid_loss += self.spectral_net_ae.SpectralNetLoss
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        spectral_valid_loss /= len(valid_loader)
        return valid_loss, spectral_valid_loss


    def _get_data_loader(self) -> tuple:
        """
        This function returns the data loaders for training, validation and testing.

        Returns:
            tuple:  The data loaders
        """
        if self.y is None:
            self.y = torch.zeros(len(self.X))
        train_size = int(0.9 * len(self.X))
        valid_size = len(self.X) - train_size
        dataset = TensorDataset(self.X, self.y)
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(
            train_dataset, batch_size = self.batch_size, shuffle = True
        )
        ortho_loader = DataLoader(
            train_dataset, batch_size = self.batch_size, shuffle = True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size = self.batch_size, shuffle = False
        )
        return train_loader, ortho_loader, valid_loader
