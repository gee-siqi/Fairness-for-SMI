import numpy as np
import pandas as pd
import json
from tools.distribution import kde_simu
import matplotlib.pyplot as plt


class Creator:
    def __init__(self, simu_cat):
        # TODO: kde method to simulate frequency data, find better way for quality distribution
        simu_freq = kde_simu(simu_cat, 3000)
        self.quality = np.random.randint(1, 101)
        self.frequency = np.random.choice(simu_freq)
        self.subscribers = 0


class User:
    def __init__(self):
        self.occupancy = 0
        self.followed_creators = []


# TODO: calculation needs further polish
def recom_prob(quality, frequency):
    return quality ** 2 * frequency


def save_info(creator_info, user_info):
    with open('creator_info.json', 'w') as f:
        json.dump(creator_info, f)

    with open('user_info.json', 'w') as f:
        json.dump(user_info, f)


class Simulation:
    def __init__(
            self,
            num_creators=1000,
            num_users=10000,
            attention_limit=120,
            simu_cat='gaming',
    ):
        self.creators = [Creator(simu_cat=simu_cat) for _ in range(num_creators)]
        print('finish init creators')
        self.users = [User() for _ in range(num_users)]
        print('finish init users')
        self.attention_limit = attention_limit

    def recommend_creator(self, user):
        recommendation_probs = [recom_prob(creator.quality, creator.frequency) for creator in self.creators]
        total_prob = sum(recommendation_probs)
        # normalize probabilities
        normalized_probs = [prob / total_prob for prob in recommendation_probs]
        recommended_creator_index = np.random.choice(len(self.creators), p=normalized_probs)
        return recommended_creator_index

    # follow the creator if user have enough time occupancy and the quality is top 3 of the followed ones
    def follow_decision(self, tolerance=3):
        for user in self.users:
            follow = 0
            # randomly recommend creators based on probability
            recommended_creator_index = self.recommend_creator(user)

            if user.occupancy <= self.attention_limit:
                followed_indices = user.followed_creators
                if len(followed_indices) < tolerance:
                    follow = 1
                else:
                    sorted_followed_quality = sorted([self.creators[i].quality for i in followed_indices],
                                                     reverse=True)
                    if self.creators[recommended_creator_index].quality >= sorted_followed_quality[2]:
                        follow = 1
            if follow == 1:
                user.followed_creators.append(recommended_creator_index)
                user.occupancy += self.creators[recommended_creator_index].frequency
                self.creators[recommended_creator_index].subscribers += 1

    def simulate_until_condition(self):
        i = 0
        while not all(user.occupancy > self.attention_limit for user in self.users):
            i += 1
            self.follow_decision()
            print(f'the {i} th run')
        creator_res = [{"quality": creator.quality,
                        "subscribers": creator.subscribers,
                        "frequency": creator.frequency,
                        }
                       for creator in self.creators
                       ]

        user_res = [{"occupancy": user.occupancy,
                     "followed_creators": user.followed_creators
                     }
                    for user in self.users
                    ]
        save_info(creator_res, user_res)

    def plot_creator(self):

        qualities = [creator.quality for creator in self.creators]
        subscribers = [creator.subscribers for creator in self.creators]

        plt.figure(figsize=(10, 6))
        plt.scatter(qualities, subscribers, alpha=0.5)
        plt.xlabel("Quality")
        plt.ylabel("Subscribers")
        plt.title("Creator Quality vs Subscribers")
        plt.grid()

        # 保存散点图为图像文件
        plt.savefig("creator_scatter_plot.png")

        plt.show()


# 创建模拟对象并运行模拟
simulation = Simulation()
simulation.simulate_until_condition()
# 绘制散点图并保存
simulation.plot_creator()
