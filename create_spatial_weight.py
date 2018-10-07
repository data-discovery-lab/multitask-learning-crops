import csv
import pandas as pd
import numpy as np

class Neighbor:
    def __init__(self, id, distance, crop_yield):
        self.id = id
        self.distance = distance
        self.weight = 1
        self.power_factor = 2
        self.crop_yield = crop_yield

    def get_id(self):
        return self.id

    def compute_weight(self, group_size):
        self.weight = 1 / (group_size * (self.distance**self.power_factor))

    def get_weight(self):
        return self.weight

    def get_crop_yield(self):
        return self.crop_yield

    def get_distance(self):
        return self.distance


class Group:
    def __init__(self, id, crop_yield):
        self.id = id
        self.crop_yield = crop_yield
        self.neighbors = []
        self.weight = 0

    def add_neighbor(self, nb):
        self.neighbors.append(nb)

    def get_crop_yield(self):
        return self.crop_yield

    def get_neighbors(self):
        return self.neighbors

    def get_id(self):
        return self.id

    def get_size(self):
        return len(self.neighbors)

    def compute_neighbor_weight(self):
        size = self.get_size()
        w = 0
        for nb in self.neighbors:
            nb.compute_weight(size)
            # w_operand = nb.get_weight()*(abs(self.crop_yield - nb.get_crop_yield())**2)
            # w_operand = w_operand / self.crop_yield
            # w = w + w_operand

        # self.weight = w
    #
    # def get_weight(self):
    #     return self.weight

groups = {}
region_neighbors = {}

nbh_distance = 1
df = pd.read_csv('data/nb_size_' + str(nbh_distance) + '.csv')


for index, row in df.iterrows():

    current = row['source']
    neighbor = row['neighbor']
    distance = row['distance']
    s_yield_00 = row['s_y00']
    s_yield_01 = row['s_y01']
    s_yield_02 = row['s_y02']
    s_yield_03 = row['s_y03']
    nb_yield_00 = row['nb_y00']
    nb_yield_01 = row['nb_y01']
    nb_yield_02 = row['nb_y02']
    nb_yield_03 = row['nb_y03']

    if current not in region_neighbors:
        region_neighbors[current] = []

    nb = Neighbor(id=neighbor, distance=distance, crop_yield=[nb_yield_00, nb_yield_01, nb_yield_02, nb_yield_03])
    ## append neighbor list
    region_neighbors[current].append(nb)

    if current not in groups:
        groups[current] = Group(id=current, crop_yield=[s_yield_00, s_yield_01, s_yield_02, s_yield_03])

    g = groups[current]
    g.add_neighbor(nb)


# compute normalized weight and write to file
with open('data/nb_size_' + str(nbh_distance) + '_weight.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["source", "s_y00", "s_y01", "s_y02", "s_y03", "nb_size", "neighbor", "distance", "nb_weight", "nb_y00", "nb_y01", "nb_y02", "nb_y03"])

    for g_id, g in groups.items():
        g.compute_neighbor_weight()

        for nb in g.get_neighbors():
            print('id:', g_id, 'nb:', nb.get_id(), 'weight:', nb.get_weight())
            line = np.concatenate((g_id, g.get_crop_yield(), g.get_size(),
                                  nb.get_id(), nb.get_distance(), nb.get_weight(), nb.get_crop_yield()), axis=None)
            writer.writerow(line)


# compute normalized weight and write to file
# with open('data/finalized_weight.csv', 'w') as csv_file:
#     writer = csv.writer(csv_file, delimiter=',')
#     writer.writerow(["source", "nb_weight"])
#
#     for g_id, g in groups.items():
#         g.compute_neighbor_weight()
#
#         line = [g_id, g.get_weight()]
#         writer.writerow(line)
