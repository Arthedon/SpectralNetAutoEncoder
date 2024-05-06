from utils import load_data
from spectralnet import SpectralNet
from SpectralNetAE import SpectralNetAE
from spectralnet._metrics import Metrics 

from matplotlib import pyplot as plt
import torch
import numpy as np
import pandas as pd

def main():
    x_train, x_test, y_train, y_test = load_data("two_moons")

    X = torch.cat([x_train, x_test])
    y = torch.cat([y_train, y_test])


    print(X[0].shape)

    spectralnetae = SpectralNetAE(
        n_clusters = 2,
        use_ae = False,
        use_siamese = False,
        spectral_batch_size = 718,
        spectral_max_epochs = 1000,
        spectral_tolerance  = 1e-9,
        spectral_is_local_scale = False,
        spectral_num_neighbours = 8,
        spectral_scale_k = 2,
        spectral_lr = 1e-2,
        spectral_config = {"hiddens": [128, 128, 2]},
        spectral_is_normalized = False,
        spectral_alpha =  2e-3,
        spectral_conv = False
    )

    spectralnet = SpectralNet(
        n_clusters = 2,
        should_use_ae = False,
        should_use_siamese = False,
        spectral_batch_size = 718,
        spectral_epochs = 40,
        spectral_is_local_scale = False,
        spectral_n_nbg = 8,
        spectral_scale_k = 2,
        spectral_lr = 1e-2,
        spectral_hiddens = [128, 128, 2],
    )

    train_loss_count, valid_loss_count, spectral_train_loss_count, spectral_valid_loss_count = spectralnetae.fit(X, y)
    ae_cluster_assignments = spectralnetae.predict_clusters(X)
    ae_embeddings = spectralnetae.embeddings_
    x_estimate = spectralnetae.predict(X)

    spectralnet.fit(X, y)
    cluster_assignments = spectralnet.predict(X)
    embeddings = spectralnet.embeddings_

    if y is not None:
        print("SpectralNetAE :")
        y = y.detach().cpu().numpy()
        acc_score = Metrics.acc_score(ae_cluster_assignments, y, n_clusters=2)
        nmi_score = Metrics.nmi_score(ae_cluster_assignments, y)
        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")

    if y is not None:
        print("SpectralNet :")
        acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters=2)
        nmi_score = Metrics.nmi_score(cluster_assignments, y)
        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")


    return x_estimate, ae_embeddings, ae_cluster_assignments, X.detach().cpu().numpy(), y, train_loss_count, valid_loss_count, spectral_train_loss_count, spectral_valid_loss_count, embeddings, cluster_assignments


if __name__ == "__main__":
    x_estimate, ae_embeddings, ae_assignments, X, y, train_loss_count, valid_loss_count, spectral_train_loss_count, spectral_valid_loss_count, embeddings, assignments = main()
    
fig = plt.figure(figsize = (12, 12))
ax1 = fig.add_subplot(331)
ax1.scatter(X[:, 0], X[:, 1], c = ae_assignments)
ax1.title.set_text('SpectralNetAE Clusters assignments')

ax2 = fig.add_subplot(332)
ax2.scatter(X[:, 0], X[:, 1], c = assignments)
ax2.title.set_text('SpectralNet Clusters assignments')

ax3 = fig.add_subplot(333)
ax3.scatter(X[:, 0], X[:, 1], c = y)
ax3.title.set_text('Real classes')

ax4 = fig.add_subplot(334)
ax4.scatter(ae_embeddings[:, 0], ae_embeddings[:, 1], c = y)
ax4.title.set_text('SpectralNetAE Embedding')

ax5 = fig.add_subplot(335)
ax5.scatter(ae_embeddings[:, 0], embeddings[:, 1], c = y)
ax5.title.set_text('SpectralNet Embedding')

ax6 = fig.add_subplot(336)
ax6.scatter(x_estimate[:, 0].detach().cpu(), x_estimate[:, 1].detach().cpu(), c = y)
ax6.title.set_text('Reproduction')

ax7 = fig.add_subplot(337)
ax7.plot(spectral_train_loss_count, c = 'b')
ax7.plot(spectral_valid_loss_count, c = 'r')
ax7.legend(['Train', 'Valid'])
ax7.title.set_text('Spectral Loss')

ax8 = fig.add_subplot(338)
ax8.plot(train_loss_count, c = 'b')
ax8.plot(valid_loss_count, c = 'r')
ax8.legend(['Train', 'Valid'])
ax8.title.set_text('Loss')

plt.show()