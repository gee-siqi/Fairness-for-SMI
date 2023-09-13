import numpy as np
from tools.utils import get_root_path, path
import concurrent.futures


# from simu_process.distribution import kde_simu


class Creator:
    def __init__(self, config, id):
        """
        The distribution of frequency is simulated by uploading frequency of top 1000 YouTubers in specific category.
        param simu_cat: the category we want to simulate frequency distribution,
        id: ranking,

        """
        self.id = id
        self.subscribers = 0
        self.views = 0

        # get distribution of frequency of a given category
        root_dir = get_root_path()
        data_dir = path(root_dir, f'simu_process/freq_simu_{config["simu_cat"]}.npy')
        simu_freq = np.load(data_dir)

        # if the frequency is more than 200, re-choose frequency
        self.frequency = 1000
        while self.frequency > 200 or self.frequency < 0:
            self.frequency = np.random.choice(simu_freq)
        self.contents = self.frequency


class User:
    """
    define users
    """

    def __init__(self, config, id):
        self.config = config
        self.id = id
        self.followed_creators = []
        self.consume = 0
        # self.searching_time = 0
        self.finish_time = None

    def decision(self, cc):
        """
        Users follow top n CCs among the followed ones.
        param cc: the one recommended to the user at this step
        :return: whether to follow the cc
        """

        follow = False
        if self.finish_time is None and cc.id not in self.followed_creators:
            # if len(self.followed_creators) < self.config["tolerance"]:
            #     follow = True
            if cc.id < sorted(self.followed_creators)[self.config["tolerance"] - 1]:
                follow = True
        return follow


class Network:
    def __init__(self, config, G=None):

        self.config = config
        self.tolerance = config["tolerance"]
        self.attention_limit = config["attention_limit"]
        num_creators = config["num_CCs"]
        num_users = config["num_users"]

        self.G = G
        if not self.G:
            self.G = np.zeros((num_users, num_creators))

    def follow(self, user, cc, step):
        """
        when user decides to follow the creator, how the network work
        :param step:
        :param user:
        :param cc:
        :return:
        """
        # attention_limit = self.config["attention_limit"]
        if self.G[user.id, cc.id] == 0 and user.finish_time is None:
            self.G[user.id, cc.id] = 1
            # assign # of contents to consume and the user consumption
            # follow = {'id': cc.id, 'contents': cc.frequency, 'frequency': cc.frequency, 'consume': 1}
            follow = {'id': cc.id, 'consume': 0}
            user.followed_creators.append(follow)
            cc.subscribers += 1
            # user.consume += 1
            # cc.views += 1
            # # one more month, each user have more contents to consume
            # if user.consume % self.attention_limit == 0:
            #     user.searching_time += 1
            #     for item in user.followed_creators:
            #         item['contents'] += item['frequency']

    def consume_followed(self, user, creators, step):
        """
        For the followed creators, user consume one from their contents in each step.
        The probability to be consumed is based on their views.
        @param step:
        @param user:
        @param creators:
        @return:
        """
        if user.followed_creators:  # and user.finish_time is None:
            # followed_creator_ids = user.followed_creators
            # views = np.array([creators[i].views for i in followed_creator_ids])
            # freqs = 1 + views
            # total_freq = np.sum(freqs)
            # probs = freqs / total_freq
            #
            # chosen_creator_idx = np.random.choice(len(probs), p=probs)
            # chosen_creator_id = followed_creator_ids[chosen_creator_idx]

            res_cc = []
            for item in user.followed_creators:
                if item['consume'] != creators[item['id']].contents:
                    res_cc.append(item['id'])

            if res_cc:
                chosen_creator_id = min(res_cc)
                user.consume += 1
                creators[chosen_creator_id].views += 1

            # print(f'{step}: user {user.id} has {user.consume} consumed')
            """
            # !!! Consider attention limit, if the sum of frequency > 120, would never converge
            # *** For convenient, here only clean up the current contents
            """
            # TODO: Consider tolerance > 1
            # Check whether get the best CCs
            if any(item['id'] == 0 for item in user.followed_creators):
                user.finish_time = step


class RS:
    """
    Recommendation system with different alpha values,
    1: PA
    -1: antiPA
    0: UR
    """

    def __init__(self, config, creators):
        self.config = config
        self.num_users = config["num_users"]

    def recommend_alpha(self, creators):
        """

        :param creators: a list of creators with their info
        :return: a list of recommendation probability for each creator
        """
        alpha = self.config["alpha"]
        views = np.array([cc.views for cc in creators])
        freqs = (1.0 + views) ** alpha * (1.0 + np.array([cc.frequency for cc in creators]))

        total_freq = np.sum(freqs)
        # if total_freq == 0:
        #     probs = np.full(len(freqs), 1 / len(freqs))
        # else:
        probs = freqs / total_freq


        # assign recommended CC for each user at each step
        recommended_creator_index = np.random.choice(len(probs), size=self.num_users, p=probs)
        return recommended_creator_index

    def recommend_extreme(self, creators):
        alpha = self.config["alpha"]
        views = np.array([cc.views for cc in creators])
        # Extreme PA
        if alpha == 1000:
            max_indices = np.where(views == np.max(views))[0]
            # if more than one creator have same extrme values, random select recommendation
            if len(max_indices) == 1:
                recommended_creator_index = [max_indices[0] for _ in range(self.num_users)]
            else:
                recommended_creator_index = np.random.choice(max_indices, size=self.num_users)

        # Extreme anti PA
        elif alpha == -1000:
            # find min indexes
            min_indices = np.where(views == np.min(views))[0]
            # if more than one creator have same extrme values, random select recommendation
            if len(min_indices) == 1:
                recommended_creator_index = [min_indices[0] for _ in range(self.num_users)]
            else:
                recommended_creator_index = np.random.choice(min_indices, size=self.num_users)
        return recommended_creator_index


class Process:
    """

    """

    def __init__(self, config):
        self.config = config
        self.alpha = config["alpha"]

        self.creators = [Creator(config, id=i) for i in range(config["num_CCs"])]
        self.users = [User(config, id=i) for i in range(config["num_users"])]

        self.rs = RS(config, self.creators)
        self.network = Network(config)

        self.step = 0
        self.user_still_searching = config["num_users"]

    # def one_step(self):
    #     self.step += 1
    #
    #     # 使用 concurrent.futures 来并行化循环操作
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #         #  submit tasks
    #         user_tasks = [executor.submit(self.process_user, user) for user in self.users]
    #         # wait for all tasks finished
    #         concurrent.futures.wait(user_tasks)
    def update_contents(self, cc):
        cc['contents'] = cc['contents'] + cc['frequency']

    def one_step(self):
        self.step += 1
        # if it's a now month, each creator have new contents
        if self.step % self.config['tolerance'] == 0:
            for cc in self.creators:
                cc.contents += cc.frequency

        if self.alpha not in [-1000, 1000]:
            recom_res = self.rs.recommend_alpha(self.creators)
        else:
            recom_res = self.rs.recommend_extreme(self.creators)
        for u in self.users:
            # exploration
            self.network.follow(u, self.creators[recom_res[u.id]], self.step)
            # consume followed channels
            self.network.consume_followed(u, self.creators, self.step)

    def check_absorb(self):
        """
        Check the number of user haven't finished consuming
        @return: Whether all users stop consuming
        """
        user_still_consuming = sum(1 for user in self.users if not user.finish_time)
        max_view_id = max(self.creators, key=lambda x: x.views).id
        print(f'the {self.step} th step remains {user_still_consuming} consuming, {max_view_id} get most views')
        return user_still_consuming == 0
