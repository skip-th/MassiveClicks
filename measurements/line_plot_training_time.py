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

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=False)

    category = 'training time'
    max_sessions = [10e6, 25e6, 50e6, 75e6, 100e6, 120e6]
    gpus = ['TitanX', 'TitanX-Pascal', 'A4000', 'A6000', 'E5-2630 v3', 'EPYC 7402P', 'E5-2630 v3 (ParClick)', 'EPYC 7402P (ParClick)'][4:6]
    # Check E5-2630 v3: various..., 0 4 25 and 0 4 50 because 50 is faster than 25.
    models = ['model_0', 'model_1', 'model_2', 'model_3']
    min_nodes = 0
    max_nodes = 16
    max_session = 120e6
    uniform_y = True
    subplts = [(0,0), (0,1), (1,0), (1,1)]

    print(f'Plotting {category} of {models} with {max_sessions} sessions on a maximum of {max_nodes} nodes using {gpus} devices.')

    itr_list = [[models[i], gpu, category, subplts[i]] for gpu in gpus for i in range(len(models))]
    for model, gpu, category, (p1,p2) in itr_list:
        gpu_data = get_from_dict(model, gpu, category)
        for k, (nodes, training_time_per_session) in enumerate(sorted(list(gpu_data.items()), key=lambda tup: int(tup[0].split('nodes_')[-1]))):
            sorted_training_time_per_session = [i for i in sorted(training_time_per_session, key=lambda tup: tup[0])]
            cleaned_training_time_per_session = [i for i in sorted_training_time_per_session if i[1] != [] and i[0] <= max_session]
            training_time = [sum(i[1])/len(i[1]) for i in cleaned_training_time_per_session]
            sessions = [i[0] for i in cleaned_training_time_per_session]

            # print(f"{model},\t {gpu},\t {nodes},\t {training_time},\t {sessions},\t {sorted_training_time_per_session}")

            # if ((gpu == 'TitanX' and nodes.split('nodes_')[-1] == '16') or (gpu == 'TitanX-Pascal' and nodes.split('nodes_')[-1] == '4') or (gpu == 'A4000' and nodes.split('nodes_')[-1] == '14') or (gpu == 'A6000' and nodes.split('nodes_')[-1] == '2') or (gpu == 'E5-2630 v3' and nodes.split('nodes_')[-1] == '16') or (gpu == 'EPYC 7402P' and nodes.split('nodes_')[-1] == '16')):
            if min_nodes <= int(nodes.split('nodes_')[-1]) <= max_nodes:
                if gpu == gpus[0]:
                    color = pl.cm.Oranges(np.linspace(0,1,len(gpu_data.keys())+1))
                    marker='o'
                elif gpu == gpus[1]:
                    color = pl.cm.Blues(np.linspace(0,1,len(gpu_data.keys())+1))
                    marker='^'
                elif gpu == gpus[2]:
                    color = pl.cm.Greens(np.linspace(0,1,len(gpu_data.keys())+1))
                    marker='s'
                elif gpu == gpus[3]:
                    color = pl.cm.Purples(np.linspace(0,1,len(gpu_data.keys())+1))
                    marker='x'
                elif gpu == gpus[4]:
                    color = pl.cm.Greys(np.linspace(0,1,len(gpu_data.keys())+1))
                    marker='v'
                elif gpu == gpus[5]:
                    color = pl.cm.Reds(np.linspace(0,1,len(gpu_data.keys())+1))
                    marker='D'

                axs[p1,p2].plot(sessions, training_time,
                                label=f"{nodes.split('nodes_')[-1]}x {gpu}", color=color[k+1], marker=marker)

    fig.suptitle(category.capitalize(), fontsize=16)
    axs[0, 0].set_title('PBM')
    axs[0, 1].set_title('CCM')
    axs[1, 0].set_title('DBN')
    axs[1, 1].set_title('UBM')

    ylim_top = (int(max([ax.get_ylim()[1] for ax in axs.flat]) / 1e2) + 1) * 1e2
    for ax in axs.flat:
        ax.set(xlabel='Sessions', ylabel=f'{category.capitalize()} (s)')
        # ax.set_xlim(left=0)
        if uniform_y:
            ax.set_ylim(0, ylim_top)
        # ax.set_yscale('log')
        # ax.label_outer()

    nlegs = len(axs.flat[0].get_legend_handles_labels()[-1])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    plt.legend(loc='lower center', ncol=nlegs if nlegs < 6 else int(nlegs / 2) + 1,
        bbox_to_anchor=(-0.1, -0.4), fancybox=True)
    if uniform_y:
        plt.savefig(f'figures/{category}_{"&".join(gpus)}.png')
        print(f'Figure saved as "figures/{category}_{"&".join(gpus)}.png"')
    else:
        plt.savefig(f'figures/{category}_{"&".join(gpus)}_nonuniform.png')
        print(f'Figure saved as "figures/{category}_{"&".join(gpus)}_nonuniform.png"')
    plt.show()

# rsync -avzPh fs0:/home/sthijsse/measurements/data/ /home/skip/measurements/data/; for m in {0..3}; do cd ~/measurements/data/model_"$m"/TitanX/; read -r -d '' -a ARRAY < <(find . -type f -exec echo '{}' \;); for i in "${ARRAY[@]}"; do RES=$(tail -n -121 "$i"); echo "$RES" > "$i"; done; done;