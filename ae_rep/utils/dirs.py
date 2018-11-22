import os

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        
        if not os.path.exists(dirs):
            os.makedirs(dirs)
            os.makedirs(dirs+'plots/')
        return 0
    
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)