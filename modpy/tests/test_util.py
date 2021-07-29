import inspect


def run_unit_test(c, verbose=True):
    """
    Runs all methods in a unit test class.

    Parameters
    ----------
    c : Class
        Unit test class
    """

    if verbose:
        print('BEGIN testing {}'.format(c.__name__))

    methods = inspect.getmembers(c, predicate=inspect.isfunction)

    for (name, method) in methods:

        if verbose:
            print('TESTING: {}'.format(name))

        method(c)

    if verbose:
        print('FINISHED testing {}'.format(c.__name__))

    return True
