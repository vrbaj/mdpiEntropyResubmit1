import numpy as np


def generate_tails(data):
    """
    Generates the tails of a given data set. And and transforms it, so the location of the pareto distribution
    for the returned tail is 0.
    Args:
        data (numpy.ndarray): Data to generate the tail for.
    Yields:
        ndarray: The next tail
    """
    for i in range(1, data.size - 1):
        yield data[:i] - data[i]


data = generate_tails(np.array([1, 2, 3, 4]))
for item in data:
    print(item)