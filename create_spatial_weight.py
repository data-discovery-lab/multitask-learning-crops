import csv

class Neighbor:
    def __init__(self, id, distance):
        self.id = id
        self.distance = distance
        self.weight = 1
        self.power_factor = 2

    def get_id(self):
        return self.id

    def compute_weight(self, group_size):
        self.weight = 1 / (group_size * self.distance**self.power_factor)

    def get_weight(self):
        return self.weight

class Group:
    def __init__(self, id):
        self.id = id
        self.neighbors = []

    def add_neighbor(self, nb):
        self.neighbors.append(nb)

    def get_neighbors(self):
        return self.neighbors

    def get_id(self):
        return self.id

    def get_size(self):
        return len(self.neighbors)

    def compute_neighbor_weight(self):
        size = self.get_size()
        for nb in self.neighbors:
            nb.compute_weight(size)



groups = {}
region_neighbors = {}
with open('data/neighbor_cell_size_1.csv') as fp:
    header = True
    distance = 1

    # compute neighbor size of each region
    for line in fp:
        if header == True:
            header = False
            continue

        regions = line.split(",")
        if len(regions) < 2:
            print("bad region, no neighbor")
            continue
        current = regions[0].strip()

        if current not in region_neighbors:
            region_neighbors[current] = []

        nb = Neighbor(regions[1].strip(), distance)
        ## append neighbor list
        region_neighbors[current].append(nb)

        if current not in groups:
            groups[current] = Group(current)

        g = groups[current]
        g.add_neighbor(nb)

# compute normalized weight and write to file
with open('data/neighbor_weight.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(["Cell", "Neighbor", "Weight"])

    for g_id, g in groups.items():
        g.compute_neighbor_weight()

        for nb in g.get_neighbors():
            print('id:', g_id, 'nb:', nb.get_id(),'weight:', nb.get_weight())
            line = [g_id,  nb.get_id(), nb.get_weight()]
            writer.writerow(line)

