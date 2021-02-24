import os
from optimization import solspace

log_dir = os.path.dirname(os.path.abspath(solspace.__file__))
log_fpath = os.path.join(log_dir, 'logs', 'optimizer.log')

def logconfiguration():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s | %(name)-2s | %(levelname)s | Module: %(filename)s |  Function: %(funcName)s | %(message)s',
        filename=log_fpath)
    
    logging = logging.getLogger()
    logging.name='OM'
    
    return logging

