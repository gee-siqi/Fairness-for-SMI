import os


def get_root_path():
    script_path = os.path.abspath(__file__)
    root_directory = os.path.dirname(os.path.dirname(script_path))
    return root_directory


def path(path1, path2):
    full_path = os.path.join(path1, path2)
    return full_path


