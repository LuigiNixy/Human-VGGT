from torch import nn


def linear_projector(student_channels: int, teacher_channels: int) -> nn.Linear:
    """Map the final channel dimension of sampled student features."""
    return nn.Linear(student_channels, teacher_channels)
