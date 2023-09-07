import numpy as np
from tools.utils import get_root_path, path

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


class User:
    """
    define users
    """

    def __init__(self, config, id):
        self.config = config
        self.id = id
        self.occupancy = 0
        self.followed_creators = []
        self.consume = 0
        self.finish_time = None

    def decision(self, cc):
        """
        When users haven't reached the attention limit, they follow top n CC among the followed ones.

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
        if self.G[user.id][cc.id] == 0 and user.finish_time is None:
            self.G[user.id][cc.id] = 1
            user.followed_creators.append(cc.id)
            # user.occupancy += cc.frequency
            user.consume += 1
            cc.subscribers += 1
            cc.views += 1
            # if out of user's attention limit, stop searching
            if user.consume > self.attention_limit:  # or all(i in user.followed_creators for i in range(self.tolerance)):
                user.finish_time = step

    def consume_followed(self, user, creators, step):
        """
        For the followed creators, user consume one from their contents in each step.
        The probability to be consumed is based on their views.
        @param step:
        @param user:
        @param creators:
        @return:
        """
        if user.followed_creators != [] and user.finish_time is None:
            views = [creators[i].views for i in user.followed_creators]
            freqs = [1+v for v in views]
            probs = [f/sum(freqs) for f in freqs]
            idx = np.random.choice(len(probs), p=probs)
            view_cc = user.followed_creators[idx]
            user.consume += 1
            creators[view_cc].views += 1
            if user.consume > self.attention_limit:
                user.finish_time = step
        # print(f'{step}: user {user.id} has {user.consume} consumed')

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
        probs = [(1 + cc.views) ** alpha * cc.frequency for cc in creators]

        total_prob = sum(probs)
        if total_prob == 0:
            probs = [1 / len(probs) for _ in probs]  # 如果总概率为零，分配均等概率
        else:
            probs /= total_prob

        # assign recommended CC for each user at each step
        recommended_creator_index = [np.random.choice(len(probs), p=probs) for _ in range(self.num_users)]
        return recommended_creator_index

    def recmd_followed(self, users, creators):
        # TODO: use alpha method to recommend contents from followed creators
        pass


class Process:
    """

    """
    def __init__(self, config):
        self.config = config

        self.creators = [Creator(config, id=i) for i in range(config["num_CCs"])]
        self.users = [User(config, id=i) for i in range(config["num_users"])]

        self.rs = RS(config, self.creators)
        self.network = Network(config)

        self.step = 0
        self.user_still_searching = config["num_users"]

    def one_step(self):
        self.step += 1
        # each user get a recommendation
        recom_res = self.rs.recommend_alpha(self.creators)

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
        # print(f'the {self.step} th step remains {user_still_consuming} consuming')
        return user_still_consuming == 0

