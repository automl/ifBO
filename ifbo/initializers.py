from collections.abc import Callable

from torch import nn


def get_NormalInitializer(std: float) -> Callable[[nn.Module], nn.Module]:
    def initializer(m: nn.Module) -> nn.Module:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, std)
            nn.init.normal_(m.bias, 0, std)

    return initializer
