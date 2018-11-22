#!/usr/bin/env ipython
import json

def hinted_tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj
    
def load_settings_from_file(identifier):
    """
    Handle loading settings from a JSON file, filling in missing settings from
    the command line defaults, but otherwise overwriting them.
    """
    settings_path = './settings/' + identifier + '.txt'
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r'), object_hook=hinted_tuple_hook)
    # check for settings missing in file
    return settings_loaded
