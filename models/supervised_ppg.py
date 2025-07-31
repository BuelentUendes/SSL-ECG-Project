import torch
import torch.nn as nn

class DilatedCNN(nn.Module):
    """
    Dilated CNN for ECG-based stress detection as specified in the architecture.
    Architecture based on: https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/wss2.70004

    Architecture:
    - Input: (batch_size, 1, 640) - 1D ECG signal
    - 8 Dilated Conv1D layers with increasing filters and dilation rates
    - Batch normalization and dropout after each conv layer
    - Global max pooling
    - Dense output layer with sigmoid activation
    """

    def __init__(self):
        super(DilatedCNN, self).__init__()

        # Architecture parameters
        self.num_filters = [16, 32, 64, 96, 128, 256, 320, 512]
        self.kernel_size = 8
        self.dilation_rates = [1, 2, 4, 8, 16, 32, 64, 128]
        self.dropout_rate = 0.5
        self.num_classes = 2

        # Build the convolutional layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # Input has 1 channel (single ECG signal)
        in_channels = 1

        for i in range(8):  # 8 convolutional layers
            # Dilated 1D Convolution
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.num_filters[i],
                kernel_size=self.kernel_size,
                dilation=self.dilation_rates[i],
                padding='same'  # Keep same length
            )

            # Batch Normalization
            bn = nn.BatchNorm1d(self.num_filters[i])

            # Dropout
            dropout = nn.Dropout(self.dropout_rate)

            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
            self.dropout_layers.append(dropout)

            # Update input channels for next layer
            in_channels = self.num_filters[i]

        # ReLU activation
        self.relu = nn.ReLU()

        # Global Max Pooling 1D
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Dense output layer
        self.output_layer = nn.Linear(self.num_filters[-1], self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, 640)

        Returns:
            Output tensor of shape (batch_size, 2) with sigmoid activation
        """
        # Pass through all 8 convolutional blocks
        for i in range(8):
            # Conv1D -> BatchNorm -> ReLU -> Dropout
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.relu(x)
            x = self.dropout_layers[i](x)

        # Global Max Pooling: (batch_size, 512, seq_len) -> (batch_size, 512, 1)
        x = self.global_max_pool(x)

        # Flatten: (batch_size, 512, 1) -> (batch_size, 512)
        x = x.squeeze(-1)

        # Dense output layer with sigmoid activation
        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x