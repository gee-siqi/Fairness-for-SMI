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
        self.month_views = 0

        # get distribution of frequency of a given category
        root_dir = get_root_path()
        data_dir = path(root_dir, f'simu_process/freq_simu_{config["simu_cat"]}.npy')
        simu_freq = np.load(data_dir)
        simu_freq = simu_freq[simu_freq > 0]
        simu_freq = simu_freq[simu_freq < 200]

        # if the frequency is more than 200, re-choose frequency
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
        # time when user finds the best CC
        self.finish_time = None

    def decision(self, cc):
        """
        Users follow top n CCs among the followed ones.
        param cc: the one recommended to the user at this step
        :return: whether to follow the cc
        """

        follow = False
        if self.finish_time is None and cc.id not in self.followed_creators:
            if len(self.followed_creators) < self.config["tolerance"]:
                follow = True
            elif cc.id < sorted([i['id'] for i in self.followed_creators])[self.config["tolerance"] - 1]:
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
        :param user: one user
        :param cc: cc recommended to the user
        """
        # if the user decides to follow and hasn't followed the CC before
        if user.decision(cc) and self.G[user.id, cc.id] == 0:
            self.G[user.id, cc.id] = 1
            # assign user consumption of this cc
            follow = {'id': cc.id, 'consume': 0}
            user.followed_creators.append(follow)
            cc.subscribers += 1
            # if the user find the best one, mark the step
            if cc.id == 0:
                user.finish_time = step

    def consume_followed(self, user, creators):
        """
        For the followed creators, user consume one from their contents in each step.
        The probability to be consumed is based on their views.
        @param step:
        @param user:
        @param creators:
        @return:
        """
        if user.followed_creators:
            # collect the followed creators not are fully consumed by the user
            res_cc = []
            for item in user.followed_creators:
                # number of views for each followed < cc's content, here assume user views one content only once.
                if item['consume'] != creators[item['id']].contents:
                    res_cc.append(item['id'])

            # only consume the contents of the best CC
            if res_cc:
                chosen_creator_id = min(res_cc)
                user.consume += 1
                creators[chosen_creator_id].views += 1
                creators[chosen_creator_id].month_views += 1

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
        self.rs_basic = config["rs_basic"]
        self.alpha = config["alpha"]

    def recommend_alpha(self, creators):
        """

        :param creators: a list of creators with their info
        :return: a list of recommendation probability for each creator
        """
        month_views = np.array([cc.month_views for cc in creators])
        upload_freqs = np.array([cc.frequency for cc in creators])
        contents = np.array([cc.contents for cc in creators])
        views = np.array([cc.views for cc in creators])

        avg_views = np.divide(month_views, (1.0 + upload_freqs))
        subscribers = np.array([cc.subscribers for cc in creators])
        total_avg_views = np.divide(views, (1.0 + contents))

        # Recommending CCs based on tatal views combined with frequency
        if self.rs_basic == 'total_views':
            freqs = (1.0 + views) ** self.alpha * (1.0 + upload_freqs)
        # Recommending CCs based on average views of this month combined with frequency
        elif self.rs_basic == 'avg_views':
            freqs = (1.0 + avg_views) ** self.alpha * (1.0 + upload_freqs)
        elif self.rs_basic == 'total_avg_views':
            freqs = (1.0 + total_avg_views) ** self.alpha * (1.0 + upload_freqs)
        # Recommending CCs based on subscribers only
        elif self.rs_basic == 'subscribers':
            freqs = (1.0 + subscribers) ** self.alpha

        else:
            raise 'RS should be based on total_views, avg_views, total_avg_views subscribers'

        freqs[np.isinf(freqs)] = 10 ** 308
        total_freq = np.sum(freqs)
        probs = freqs / total_freq

        # assign recommended CC for each user at each step
        recommended_creator_index = np.random.choice(len(probs), size=self.num_users, p=probs)
        return recommended_creator_index

    def recommend_extreme(self, creators):

        month_views = np.array([cc.month_views for cc in creators])
        upload_freqs = np.array([cc.frequency for cc in creators])
        contents = np.array([cc.contents for cc in creators])
        views = np.array([cc.views for cc in creators])

        avg_views = np.divide(month_views, (1.0 + upload_freqs))
        subscribers = np.array([cc.subscribers for cc in creators])
        total_avg_views = np.divide(views, (1.0 + contents))

        # Recommending CCs based on tatal views combined with frequency
        if self.rs_basic == 'total_views':
            freqs = (1.0 + views)
        # Recommending CCs based on average views of this month combined with frequency
        elif self.rs_basic == 'avg_views':
            freqs = (1.0 + avg_views)
        elif self.rs_basic == 'total_avg_views':
            freqs = (1.0 + total_avg_views)
        # Recommending CCs based on subscribers only
        elif self.rs_basic == 'subscribers':
            freqs = (1.0 + subscribers)
        else:
            raise 'RS should be based on total_views, avg_views, total_avg_views subscribers'

        # Extreme PA
        if self.alpha == 1000:
            max_indices = np.where(freqs == np.max(freqs))[0]
            # if more than one creator have same extrme values, random select recommendation
            if len(max_indices) == 1:
                recommended_creator_index = [max_indices[0] for _ in range(self.num_users)]
            else:
                recommended_creator_index = np.random.choice(max_indices, size=self.num_users)

        # Extreme anti PA
        elif self.alpha == -1000:
            # find min indexes
            min_indices = np.where(freqs == np.min(freqs))[0]
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

    def update_contents(self, cc):
        cc['contents'] = cc['contents'] + cc['frequency']

    def one_step(self):
        self.step += 1
        # if it's a new month, each creator have new contents, views count from 0
        if self.step % self.config['attention_limit'] == 0:
            for cc in self.creators:
                # each month create new content and reset month_view
                cc.contents += cc.frequency
                cc.month_views = 0

        if self.alpha not in [-1000, 1000]:
            recom_res = self.rs.recommend_alpha(self.creators)
        else:
            recom_res = self.rs.recommend_extreme(self.creators)
        for u in self.users:
            if u.finish_time is None:
                # decision on whether to follow
                self.network.follow(u, self.creators[recom_res[u.id]], self.step)
            # consume followed channels, even when the user find the best cc
            self.network.consume_followed(u, self.creators)

    def check_absorb(self):
        """
        Check the number of user haven't finished consuming
        @return: Whether all users stop consuming
        """
        user_still_consuming = sum(1 for user in self.users if not user.finish_time)

        # for debug
        # max_view_id = max(self.creators, key=lambda x: x.views).id
        # max_sub_id = max(self.creators, key=lambda x: x.subscribers).id
        # print(f'the {self.step} th step remains {user_still_consuming} consuming, {max_view_id} get most views, {max_sub_id} get most followers')

        return user_still_consuming == 0
