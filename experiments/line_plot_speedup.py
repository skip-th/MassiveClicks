from data_read import read_dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
import seaborn as sns
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# plt.rcParams.update({'font.size': 12})
plt.rcParams['text.usetex'] = True

def add_to_dict(dictionary, key, value):
    if key in dictionary.keys():
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]


def get_from_dict(model_type, gpu_type, category):
    time_per_nodes = dict()
    for top in data:
        for model in data[top]:
            if model == model_type:
                for gpu in data[top][model]:
                    if gpu == gpu_type:
                        for nodes in data[top][model][gpu]:
                            for sessions in data[top][model][gpu][nodes]:
                                for partitioning in data[top][model][gpu][nodes][sessions]:
                                    add_to_dict(time_per_nodes, nodes, (int(sessions.split('sessions_')[-1]), data[top][model][gpu][nodes][sessions][partitioning][category]))
    return time_per_nodes


if __name__ == '__main__':
    dataset = read_dataset()
    data = dataset.get_dataset()

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=False)

    category = 'training time'
    # parclick_training_time_PBM = np.array([98.2428, 234.887, 531.533]) # PBM: E5-2630 v3 (ParClick)
    # parclick_training_time_CCM = np.array([1278.95, 3212.03, 6523.38]) # CCM: E5-2630 v3 (ParClick)
    parclick_training_time_PBM = np.array([40.7932, 110.883, 211.095]) # PBM: EPYC 7402P (ParClick)
    parclick_training_time_CCM = np.array([448.389, 1097.06, 2114.57]) # CCM: EPYC 7402P (ParClick)
    max_sessions = [10e6, 25e6, 50e6]
    gpus = ['A4000', 'A6000']
    models = ['model_0', 'model_1']
    subplts = [0, 1]


    gpu1_colors = colors = pl.cm.Oranges(np.linspace(0,1,5+1))
    gpu2_colors = colors = pl.cm.Blues(np.linspace(0,1,3+1))
    itr_list = [[models[i], gpu, category, subplts[i]] for gpu in gpus for i in range(len(models))]
    for model, gpu, category, p1 in itr_list:
        gpu_data = get_from_dict(model, gpu, category)
        comp_data = get_from_dict(model, gpu, 'computation time')
        itr_data = get_from_dict(model, gpu, 'iteration time')
        for k, (nodes, training_time_per_session) in enumerate(sorted(list(gpu_data.items()), key=lambda tup: int(tup[0].split('nodes_')[-1]))):
            training_time = [sum(i[1])/len(i[1]) for i in sorted(training_time_per_session, key=lambda tup: tup[0]) if i[1] != []][:len(max_sessions)]
            sessions = max_sessions[:len(training_time)]

            comp_time = [sum(i[1])/len(i[1]) for i in sorted(comp_data[nodes], key=lambda tup: tup[0]) if i[1] != []][:len(max_sessions)]
            itr_time = [sum(i[1])/len(i[1]) for i in sorted(itr_data[nodes], key=lambda tup: tup[0]) if i[1] != []][:len(max_sessions)]
            ace = [comp_time[i]/itr_time[i]*100 for i in range(len(comp_time))] # percentage of time spend in computation per iteration

            comparison_data = parclick_training_time_PBM if model == 'model_0' else parclick_training_time_CCM
            speedup = comparison_data/np.array(training_time)
            if gpu == gpus[0]:
                axs[p1].plot(sessions, speedup,
                             label=f"{nodes.split('nodes_')[-1]} {gpu}", color=gpu1_colors[k+1], marker='o')
                print(F"{gpu} - {model}")
                print("Nodes | Sessions | Training Time | Speedup | ACE")
                for i in range(len(sessions)):
                    print("{}\t{}\t{}\t{}\t{}".format(nodes.split('nodes_')[-1], round(sessions[i]), round(training_time[i], 1), round(speedup[i], 1), round(ace[i], 1)), end='\n')
            elif gpu == gpus[1]:
                axs[p1].plot(sessions, speedup,
                             label=f"{nodes.split('nodes_')[-1]} {gpu}", color=gpu2_colors[k+1], marker='x')
                print(F"{gpu} - {model}")
                print("Nodes | Training Time | Speedup | ACE")
                for i in range(len(sessions)):
                    print("{}\t{}\t{}\t{}\t{}".format(nodes.split('nodes_')[-1], round(sessions[i]), round(training_time[i], 1), round(speedup[i], 1), round(ace[i], 1)), end='\n')

    fig.suptitle('Speedup ' + category, fontsize=16)
    axs[0].set_title('PBM')
    axs[1].set_title('CCM')

    # ylim_top = (int(max([ax.get_ylim()[1] for ax in axs.flat]) / 1e2) + 1) * 1e2
    for ax in axs.flat:
        ax.set(xlabel='Sessions', ylabel=f'Speedup {category}')
        ax.set_xlim(0, 55e6)
        ax.set_ylim(bottom=0)
        ax.hlines(1, linestyles='dashed', xmin=0, xmax=55e6, color='lightgray')
        # ax.label_outer()

    nlegs = len(axs.flat[0].get_legend_handles_labels()[-1])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    plt.legend(loc='lower center', ncol=nlegs if nlegs < 6 else int(nlegs / 2),
        bbox_to_anchor=(0.5, -0.4), fancybox=True)
    plt.savefig(f'figures/speedup_{category}_{"&".join(gpus)}.png')
    plt.show()
