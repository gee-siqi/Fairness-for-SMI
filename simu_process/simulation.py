import pandas as pd
import yaml
from simu_process import model
from tools.utils import get_root_path, path
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
        self.iteration = self.config["iterations"]

    def simulate(self):
        """
        run the simulate process
        @return: three lists of dict data about users and CCs after iteration
        """
        # TODO: set random seeds
        p = model.Process(self.config)
        # steps needed to get absorb
        step = 0
        while not p.check_absorb():
            step += 1
            p.one_step()

        # TODO: make it possible to start iteration from given step
        user_data = [{
            'id': u.id, 'followed': u.followed_creators, 'steps': u.finish_time, 'occupancy': u.occupancy
        } for u in p.users]
        cc_data = [{'id': cc.id, 'subs': cc.subscribers, 'freq': cc.frequency} for cc in p.creators]
        return user_data, cc_data

    def iterations(self):
        self.iter_res = []
        user_dfs = []
        cc_dfs = []

        for i in tqdm(range(self.iteration), desc=f"Iterations with alpha = {self.config['alpha']}"):
            user_data, cc_data = self.simulate()
            df_u = pd.DataFrame(user_data)
            df_c = pd.DataFrame(cc_data)
            df_u['iteration'] = i
            df_c['iteration'] = i
            user_dfs.append(df_u)
            cc_dfs.append(df_c)

        user_df = pd.concat(user_dfs)
        cc_df = pd.concat(cc_dfs)
        self.iter_res = [user_df, cc_df]

    def save_data(self, overwrite=True):
        # get the direction of output file, save json for each alpha value
        root_dir = get_root_path()
        time = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
        self.output_file = path(root_dir, f'_output/{time}alpha{self.config["alpha"]}.pkl')
        with open(self.output_file, 'wb') as file:
            pickle.dump(self.iter_res, file)

    def run_and_save(self):
        self.iterations()
        self.save_data()


def run_simu(file='~/config/config.yaml'):
    sim = Simulation(file)
    sim.simulate()
