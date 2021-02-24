"""Debug scripts.
"""


class Debugger(object):
    """Wrapper class for defining debugger"""

    enabled = False
    """
    Debugger will be activated only if parameter is set to True. Set to False by default" \
    """

    # func is the function to be wrapped
    def __init__(self, func):
        self.func = func

    # Custom call method
    def __call__(self, *args, **kwargs):
        if self.enabled:
            print('Entering', self.func.func_name)
            [print('    arg:', kwarg) for kwarg in kwargs if kwarg == 'item_id']
        return self.func(*args, **kwargs)
