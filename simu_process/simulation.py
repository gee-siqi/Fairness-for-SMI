import os
import pandas as pd
import numpy as np
import yaml
from simu_process import model
from tools.utils import get_root_path
import pickle
from tqdm import tqdm
from datetime import datetime
import os


class Simulation:
    def __init__(self, file_path):  # 'config.yaml'
        """

        @param file_path: the full config file name
        """
        self.iter_res = None
        self.output_file = None
        with open(file_path, 'r') as file:
            self.config = yaml.safe_load(file)
        # print('load yaml finished')
        self.runs = self.config["runs"]
        self.time_constrain = self.config["time_constrain"]

    def simulate(self, random_seed):
        """
        run the simulate process
        @return: three lists of dict data about users and CCs after iteration
        """
        np.random.seed(random_seed)
        p = model.Process(self.config)
        step = 0
        # constrain the step of the simulation
        while step < self.time_constrain:
            step += 1
            p.one_step()
            # p.check_absorb()
            # in validation mode, each iteration stops when all users get the best one
            if self.config["rs_basic"] == 'subscribers' and p.check_absorb():
                break

        user_data = [{
            'id': u.id, 'steps': u.finish_time
        } for u in p.users]
        cc_data = [{'id': cc.id, 'subs': cc.subscribers, 'view': cc.views, 'frequency': cc.frequency} for cc in p.creators]
        return user_data, cc_data

    def iterations(self):
        # generate array of random seeds for runs in each alpha
        s = np.arange(0, 1000)
        # randomly select 100 different random seeds for simulations
        # self.config['random_seed'] help to avoid same seeds array for all alphas
        rnd = np.random.RandomState(self.config['random_seed'])
        seeds = rnd.choice(s, size=self.runs, replace=False)

        self.iter_res = []
        user_dfs = []
        cc_dfs = []

        for i in tqdm(range(self.runs), desc=f"Simulations '{self.config['rs_basic']}' with alpha = {self.config['alpha']}"):
            user_data, cc_data = self.simulate(random_seed=int(seeds[i]))
            df_u = pd.DataFrame(user_data)
            df_c = pd.DataFrame(cc_data)
            df_u['runs'] = i
            df_c['runs'] = i
            user_dfs.append(df_u)
            cc_dfs.append(df_c)

        user_df = pd.concat(user_dfs)
        cc_df = pd.concat(cc_dfs)
        self.iter_res = [user_df, cc_df]

    def save_data(self, overwrite=True):
        # get the direction of output file, save json for each alpha value
        root_dir = get_root_path()
        time = datetime.strftime(datetime.now(), "%y%m%d%H%M")
        self.output_file = os.path.join(root_dir, f'_output/{self.config["rs_basic"]}_alpha{self.config["alpha"]}_{time}.pkl')
        with open(self.output_file, 'wb') as file:
            pickle.dump(self.iter_res, file)

    def run_and_save(self):
        self.iterations()
        self.save_data()


def run_simu(file='~/config/config.yaml'):
    sim = Simulation(file)
    sim.simulate()


# for test
if __name__ == '__main__':
    config_list = [
        '/Users/gee/PycharmProjects/Fairness-for-SMI/config/configalpha_-0.5.yaml',
    ]

    simu = Simulation(config_list[0])
    simu.run_and_save()
    output = simu.output_file