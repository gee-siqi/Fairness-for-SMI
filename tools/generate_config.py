import yaml
from tools.utils import get_root_path, path


def gen_config(base_file, value, value_list):
    root_dir = get_root_path()
    config_path = path(root_dir, 'config')

    # read the value of base config file
    with open(path(config_path, base_file), "r") as file:
        base_config = yaml.safe_load(file)

    # generate new config files and return the list of the file name
    config_list = []
    seed = 0
    for v in value_list:
        base_config[value] = v
        # assign different random seed for each alpha.
        base_config['random_seed'] = seed
        seed += 1
        new_config = path(config_path, f'config{value}_{v}.yaml')
        with open(new_config, 'w') as file:
            yaml.dump(base_config, file)

        config_list.append(new_config)
    return config_list
