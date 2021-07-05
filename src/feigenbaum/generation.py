from dataclasses import dataclass
from typing import Generator


@dataclass
class StableValue:
    value: float
    num_iterations: float
    x: float
    alpha: float
    precision: float


def find_stable_value(
    x: float, alpha: float, precision: float = 1e-6, max_iterations: int = 1000
) -> StableValue:
    """Find value approached by the evolution equation
    with an initial population x and growth value alpha.

    Parameters
    ----------
    x : float
        initial population ratio (0-1)
    alpha : float
        growth parameter
    precision : float
        target precision of difference between two successive values
    max_iterations : float
        number of iterations to complete before raising IndexError

    Returns
    -------
    StableValue
    """
    values = iterate(x, alpha, max_iterations)

    previous_value = next(values)

    for iteration, value in enumerate(values):
        difference = abs(value - previous_value)
        if difference <= precision:
            return StableValue(value, iteration, x, alpha, precision)
        previous_value = value

    raise IndexError(f"Maximum number of iterations ({max_iterations}) exceeded")


def iterate(
    x: float, alpha: float, num_iterations: int
) -> Generator[float, None, None]:
    x_prev = x
    for _ in range(num_iterations):
        x = alpha * x_prev * (1 - x_prev)
        yield x
        x_prev = x
