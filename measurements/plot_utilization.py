import csv
import numpy as np
import matplotlib.pyplot as plt
import datetime

def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return np.array([np.array([j.strip() for j in i]) for i in data[:-1]])

def get_categories(data):
    return np.unique(data[:,1])

def get_data_by_category(data, category):
    return np.array([i for i in data if i[1] == category])

def get_data_by_column_types(data, column_types):
    return data[:,column_types]

if __name__ == '__main__':
    path = 'utilization/' + 'TitanX-Pascal_model3_10M_itr50.csv'
    raw_data = read_csv(path)
    categories = get_categories(raw_data)
    column_types = ["device id", "category", "timestamp", "value"]

    category = categories[2]
    column_type = column_types[3]

    print(f"Categories: {categories}\nColumns: {column_types}\n" + \
          f"Currently using category \"{category}\" and column type \"{column_type}\" " + \
          f"to plot the data from file \"{path}\".")

    data_from_category = get_data_by_category(raw_data, category)
    data_value = get_data_by_column_types(data_from_category, column_types.index(column_type))
    data_timestamp = (get_data_by_column_types(data_from_category, 2)).astype(float) / 10e5

    # Plot the data
    sort_indices = np.argsort(data_timestamp)
    y = data_value[sort_indices].astype(int)
    x = (data_timestamp.astype(float) - np.min(data_timestamp.astype(float)))[sort_indices] # start at 0
    # x = (data_timestamp[sort_indices].astype(int) - np.min(data_timestamp.astype(int))) / 10e6
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.plot(x, y)
    plt.ylim(bottom=0)
    plt.xlabel('Time (s)')
    plt.ylabel(f'{category}')
    plt.title(f'{category} of {path.split("/")[1].split(".csv")[0]}')
    plt.tight_layout()
    plt.ylim(0, 100)
    plt.savefig(f'utilization/figures/{path.split("/")[1].split(".csv")[0]}.png')
    plt.show()
