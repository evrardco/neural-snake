from math import isclose
def is_opposite(alpha, beta):
    """
    returns -1 if opposite direction, otherwise 1
    """
    if isclose(abs(alpha - beta), 180.0):
        return -1
    return 0