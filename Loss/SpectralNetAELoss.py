import torch
import torch.nn as nn

class SpectralNetAELoss(nn.Module):
    def __init__(self):
        super(SpectralNetAELoss, self).__init__()

    def forward(
        self, x_in: torch.Tensor, x_out: torch.Tensor, SpectralNetLoss: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """
        This function computes the loss of the SpectralNet AutoEncoder model.
        The loss is the sum of the Mean Square Error between the input and the output and the SpectralNet Loss.

        Args:
            x_in (torch.Tensor):               Input of the network
            x_out (torch.Tensor):              Output of the network
            SpectralNetLoss (torch.Tensor):    The SpectralNetLoss at the end of the encoder
            alpha (float):                     Regularization term (importance of reconstruction in the loss computation)

        Returns:
            torch.Tensor: The loss
        """
        criterion = nn.MSELoss()
        loss = alpha * criterion(x_in, x_out) +  SpectralNetLoss

        return loss