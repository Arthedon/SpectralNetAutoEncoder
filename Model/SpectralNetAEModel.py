import torch
import numpy as np
import torch.nn as nn
from spectralnet._losses._spectralnet_loss import SpectralNetLoss
from spectralnet._utils import get_nearest_neighbors, get_gaussian_kernel, compute_scale

class SpectralNetAEModel(nn.Module):
    def __init__(self, affinity_config: dict, architecture: dict, input_dim: int, use_convolution: bool = False):

        super(SpectralNetAEModel, self).__init__()
        self.architecture = architecture
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.input_dim = input_dim
        self.scale_k = affinity_config['scale_k']
        self.num_neighbours = affinity_config['num_neighbours']
        self.is_local_scale = affinity_config['is_local_scale']
        self.device = affinity_config['device']
        self.spectral_criterion = SpectralNetLoss()
        self.use_convolution = use_convolution

        if use_convolution:
            max_pooling = self.architecture['max_pooling']
            out_channels = self.architecture['out_channels']
            image_size = self.architecture['image_size']
            kernel_sizes = self.architecture['kernel_size']
            if "stride" in self.architecture:
                strides = self.architecture['stride']
            else:
                strides = [1] * len(max_pooling)
            if "padding" in self.architecture:
                paddings = self.architecture['padding']
            else:
                paddings = [1] * len(max_pooling)
            
            # Building encoder part
            in_channel = self.input_dim
            for i in range(len(max_pooling)):
                out_channel, kernel_size, stride, padding = out_channels[i], kernel_sizes[i], strides[i], paddings[i]
                image_size = ((image_size + 2 * padding - (kernel_size - 1) - 1) / stride) + 1
                if max_pooling[i]:
                    self.encoder.append(
                        nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size = 2, stride = 2))
                    )
                    image_size = image_size // 2
                else:
                    self.encoder.append(
                        nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                                      nn.ReLU())
                    )
                in_channel = out_channel

            num_features = int(out_channel * image_size * image_size)
            num_neurons = self.architecture['num_neurons']
            num_classes = self.architecture['num_classes']
            self.encoder.append(
                nn.Sequential(nn.Flatten(), nn.Linear(num_features, num_neurons), nn.LeakyReLU())
            )
            self.encoder.append(
                nn.Sequential(nn.Linear(num_neurons, num_classes), nn.Tanh())
            )

            # Building decoder part
            self.decoder.append(
                nn.Sequential(nn.Linear(num_classes, num_neurons), nn.Tanh())
            )

            # in_channel = out_channel here but used to be clearer
            self.decoder.append(
                nn.Sequential(nn.Linear(num_neurons, num_features), nn.LeakyReLU(), nn.Unflatten(1, torch.Size([in_channel, int(image_size), int(image_size)])))
            )

            for i in reversed(range(len(max_pooling) - 1)):
                out_channel, kernel_size, stride, padding = out_channels[i], kernel_sizes[i], strides[i], paddings[i]
                if max_pooling[i + 1]:
                    self.decoder.append(
                        nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                                      nn.ReLU(),
                                      nn.UpsamplingNearest2d(scale_factor = 2))
                    )
                else:
                    self.decoder.append(
                        nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                                      nn.ReLU())
                    )
                in_channel = out_channel
            out_channel = self.input_dim
            if max_pooling[0]:
                self.decoder.append(
                    nn.Sequential(nn.UpsamplingNearest2d(scale_factor = 2), nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
                )
            else:
                self.decoder.append(
                    nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
                )

        else:
            # Building encoder part
            current_dim = self.input_dim
            for i, layer in enumerate(self.architecture['hiddens']):
                next_dim = layer
                if i == len(self.architecture['hiddens']) - 1:
                    self.encoder.append(
                        nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                    )
                else:
                    self.encoder.append(
                        nn.Sequential(nn.Linear(current_dim, next_dim), nn.LeakyReLU())
                    )
                    current_dim = next_dim

            # Building decoder part
            last_dim = self.input_dim
            current_dim = self.architecture['hiddens'][-1]
            for i, layer in enumerate(reversed(self.architecture['hiddens'][:-1])):
                next_dim = layer
                if i == 0:
                    self.decoder.append(
                       nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                    )
                else:
                    self.decoder.append(
                       nn.Sequential(nn.Linear(current_dim, next_dim), nn.LeakyReLU())
                    )
                current_dim = next_dim

            self.decoder.append(nn.Sequential(nn.Linear(current_dim, last_dim)))

    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalize the encoder of the network using the Cholesky decomposition.

        Parameters
        ----------
        Y : torch.Tensor
            The output of the encoder.

        Returns
        -------
        torch.Tensor
            The orthonormalized output of the encoder.

        Notes
        -----
        This function applies QR decomposition to orthonormalize the output (`Y`) of the encoder.
        The inverse of the R matrix is returned as the orthonormalization weights.
        """

        m = Y.shape[0]
        _, R = torch.linalg.qr(Y)
        orthonorm_weights = np.sqrt(m) * torch.inverse(R)
        return orthonorm_weights

    def encode(self, x: torch.Tensor, update_orth_weights: bool = True) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)

        Y_tilde = x
        if update_orth_weights:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)

        Y = Y_tilde @ self.orthonorm_weights

        return Y

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ torch.linalg.inv(self.orthonorm_weights)
        for layer in self.decoder:
            x = layer(x)

        return x

    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        This function computes the affinity matrix W using the Gaussian kernel.

        Args:
            X (torch.Tensor):   The input data

        Returns:
            torch.Tensor: The affinity matrix W
        """

        is_local = self.is_local_scale
        num_neighbours = self.num_neighbours
        scale_k = self.scale_k

        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k = num_neighbours + 1)
        scale = compute_scale(Dis, k = scale_k, is_local = is_local)

        W = get_gaussian_kernel(
            Dx, scale, indices, device = self.device, is_local = is_local
        )

        return W

    def spectral_loss(self, x: torch.Tensor, y: torch.Tensor, is_normalized: bool = False, siamese_net = None):
        """
        This function computes the SpectralNet loss of the encoder.
        The loss is the rayleigh quotient of the Laplacian matrix obtained from W,
        and the orthonormalized output of the network.

        Args:
            X (torch.Tensor):                         Input of the network
            Y (torch.Tensor):                         Output of the encoder
            is_normalized (bool, optional):           Whether to use the normalized Laplacian matrix or not.
            siamsese_net (SiameseNetModel, optional): Siamese Network to use for the computation of the affinity matrix

        Returns:
            torch.Tensor: The loss
        """
        if self.use_convolution:
            X = x.view(x.size(0), -1)
        else:
            X = x
        if siamese_net is not None:
            x_siamese = x.view(x.size(0), -1)
            with torch.no_grad():
                x_siamese = siamese_net.forward_once(x_siamese)
            W = self._get_affinity_matrix(x_siamese)
        else:
            W = self._get_affinity_matrix(X)

        self.SpectralNetLoss = self.spectral_criterion(W, y, is_normalized)

        return self.SpectralNetLoss

    def forward(self, X_in: torch.Tensor, is_normalized = False, siamese_net = None) -> torch.Tensor:

        Y = self.encode(X_in)

        self.spectral_loss(X_in, Y, is_normalized, siamese_net)

        X_out = self.decode(Y)

        return X_out