import pickle as pkl

default_path = 'D:\\Codings\\github_repo\\ComplexNetworkBackend\\data\\progress100.pkl'
# default_path = '../../data/progress100.pkl'
# default_path = '../../data/progress.pkl'


def load_data(path=default_path):
    with open(path, "rb+") as f:
        file_data = pkl.load(f, encoding="GBK")
    return file_data


def cal_average_path_length(path_dict):
    total_len = 0
    path_count = 0
    for source_id, paths in path_dict.items():
        for target_id, path in paths.items():
            total_len += len(path) - 1
        path_count += len(paths.keys()) - 1
    if not path_count:
        return 0
    return total_len / path_count


def get_combo_name(communities, element):
    for com in communities:
        if element in com:
            return 'Combo' + str(communities.index(com))
    return 'Default Combo'


def get_combos_arr(length):
    names = ['Combo' + str(i) for i in range(length)]
    res = [{'id': item, 'label': item} for item in names]
    return res
