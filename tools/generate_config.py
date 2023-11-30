import yaml
import os
from tools.utils import get_root_path


def gen_config(base_file, bs_values, value, value_list):
    config_list = []
    for bs in bs_values:
        root_dir = get_root_path()
        config_path = os.path.join(root_dir, 'config')

        # read the value of base config file
        with open(os.path.join(config_path, base_file), "r") as file:
            base_config = yaml.safe_load(file)

        base_config['rs_basic'] = bs
        # generate new config files and return the list of the file name
        seed = 0
        for v in value_list:
            base_config[value] = v
            # assign different random seed for each alpha.
            base_config['random_seed'] = seed
            seed += 1
            new_config = os.path.join(config_path, f'{bs}_config{value}_{v}.yaml')
            with open(new_config, 'w') as file:
                yaml.dump(base_config, file)

            config_list.append(new_config)

    return config_list
