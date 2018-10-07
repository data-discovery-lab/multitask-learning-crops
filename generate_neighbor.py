import numpy as np
import csv
import pandas as pd

class Cell:
    def __init__(self, r_id, c_id, label, active=False):
        self.r_id = r_id
        self.c_id = c_id
        self.label = label
        self.active = active

    def get_row_id(self):
        return self.r_id

    def get_col_id(self):
        return self.c_id

    def get_index(self):
        return self.r_id * 25 + self.c_id

    def is_active(self):
        return self.active

    def get_label(self):
        return self.label

class Row:
    def __init__(self, id, start_label, start_valid_label, end_valid_label):
        self.id = id
        self.start_valid_label = start_valid_label
        self.start_label = start_label
        self.end_valid_label = end_valid_label
        self.cells = np.array([])
        self.row_size = 25

    def get_id(self):
        return self.id

    def add_cell(self, cell):
        self.cells = np.append(self.cells, cell)

    def get_cells(self):
        return self.cells

    def generate_cells(self):
        for col in range(25):
            label = self.start_label + col
            active = False
            if self.id != 9 and self.id != 11 and self.id != 12 and self.id != 13:
                active = True
                if label < self.start_valid_label or label > self.end_valid_label:
                    active = False
            elif self.id == 9:
                active = True
                if col >= 1 and col <= 3:
                    active = False
            elif self.id == 11:
                active = True
                if col >= 12 and col <= 13:
                    active = False
            elif self.id == 12:
                active = True
                if col >= 11 and col <= 13:
                    active = False
                if col >= 14:
                    label = label - 1
            elif self.id == 13:
                active = True
                if col >= 12 and col <= 13:
                    active = False

            cell = Cell(self.id, col, label, active)
            self.add_cell(cell)


row_configs = [
    Row(0, 3009, 3018, 3024),
    Row(1, 3035, 3042, 3053),
    Row(2, 3067, 3072, 3086),
    Row(3, 3103, 3108, 3123),
    Row(4, 3142, 3147, 3163),
    Row(5, 3182, 3187, 3163),
    Row(6, 3225, 3230, 3248),
    Row(7, 3270, 3275, 3293),
    Row(8, 3317, 3321, 3341),
    Row(9, 3367, 3367, 3391),
    Row(10, 3421, 3421, 3445),
    Row(11, 3475, 3475, 3499),
    Row(12, 3528, 3528, 3551),
    Row(13, 3580, 3580, 3604),
    Row(14, 3634, 3634, 3658),
    Row(15, 3687, 3687, 3711),
    Row(16, 3738, 3739, 3762),
    Row(17, 3788, 3789, 3811),
    Row(18, 3837, 3839, 3860),
    Row(19, 3883, 3885, 3905),
    Row(20, 3928, 3931, 3950),
    Row(21, 3969, 3973, 3990),
    Row(22, 4006, 4011, 4026),
    Row(23, 4039, 4045, 4057),
    Row(24, 4067, 4075, 4083),
]

cell_map = dict()
## Generate data
for index, r in enumerate(row_configs):
    r.generate_cells()
    cells = r.get_cells()
    for c in cells:
        c_index = c.get_index()
        if c_index not in cell_map:
            cell_map[c_index] = c

print('cell map size', len(cell_map))
df = pd.read_csv('data/data_excel_converted.csv', delimiter=',')
cell_year_yield = dict()
for index, row in df.iterrows():
    cell_id = row['ID']
    yield_2000 = row['YLD00']
    yield_2001 = row['YLD01']
    yield_2002 = row['YLD02']
    yield_2003 = row['YLD03']

    cell_year_yield_id = str(cell_id) + 'y00'
    cell_year_yield[cell_year_yield_id] = yield_2000

    cell_year_yield_id = str(cell_id) + 'y01'
    cell_year_yield[cell_year_yield_id] = yield_2001

    cell_year_yield_id = str(cell_id) + 'y02'
    cell_year_yield[cell_year_yield_id] = yield_2002

    cell_year_yield_id = str(cell_id) + 'y03'
    cell_year_yield[cell_year_yield_id] = yield_2003

## compute neighbors size=theta
for neighbor_size in range(1, 6):

    with open('data/nb_size_' + str(neighbor_size) + '.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["source", "s_y00", "s_y01", "s_y02", "s_y03", "neighbor", "distance", "nb_y00", "nb_y01", "nb_y02", "nb_y03"])

        for r in row_configs:
            cells = r.get_cells()
            for c in cells:
                if not c.is_active():
                    continue

                r_id = c.get_row_id()
                c_id = c.get_col_id()

                nb_r_min = r_id - neighbor_size
                nb_c_min = c_id - neighbor_size
                nb_r_max = r_id + neighbor_size
                nb_c_max = c_id + neighbor_size

                source = str(c.get_label())
                s_y00 = cell_year_yield[source + 'y00']
                s_y01 = cell_year_yield[source + 'y01']
                s_y02 = cell_year_yield[source + 'y02']
                s_y03 = cell_year_yield[source + 'y03']

                for nb_r in range(nb_r_min, nb_r_max+1):
                    if nb_r < 0 or nb_r >= 25:
                        continue
                    for nb_c in range(nb_c_min, nb_c_max+1):
                        if nb_c < 0 or nb_c >= 25:
                            continue

                        if r_id == nb_r and c_id == nb_c:
                            print("same cell")
                            continue

                        nb_index = nb_r * 25 + nb_c
                        if nb_index not in cell_map:
                            print("row:", nb_r, "col:", nb_c, "is not an active cell")
                            continue
                        nb_cell = cell_map[nb_index]

                        if not nb_cell.is_active():
                            print("row:", nb_r, "col:", nb_c, "is not active")
                            continue

                        current_distance = max([abs(nb_r - r_id), abs(nb_c - c_id)])

                        nb = str(nb_cell.get_label())
                        if nb == '3025':
                            print('hi')
                        nb_y00 = cell_year_yield[nb + 'y00']
                        nb_y01 = cell_year_yield[nb + 'y01']
                        nb_y02 = cell_year_yield[nb + 'y02']
                        nb_y03 = cell_year_yield[nb + 'y03']

                        writer.writerow([source, s_y00, s_y01, s_y02, s_y03, nb, current_distance, nb_y00, nb_y01, nb_y02, nb_y03])



        ## write to file