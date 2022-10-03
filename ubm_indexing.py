
# thread_index * max_index + rank * (rank + 1) / 2 + prev_click_rank[rank]

MAX_SERP_LENGTH = 10
max_index = int(MAX_SERP_LENGTH * (MAX_SERP_LENGTH - 1) / 2 + MAX_SERP_LENGTH);

partition_size = 13
n_threads = partition_size
print(f"Queries = {partition_size}, Threads = {n_threads}, SERP length = {MAX_SERP_LENGTH}, Elements per thread = {max_index}")

itr = 0
indices = []
for rank in range(MAX_SERP_LENGTH):
    print("\nitr:   " + ''.join([f"{i:<6}" for i in range(itr, itr + rank + 1)]) + "\nranks: " + ''.join([f"{r:<6}" for r in range(rank + 1)]) + "\n" + "=" * 5 + "=" * (rank + 1) * 6)
    itr += rank + 1
    for thread_index in range(n_threads):
        string = f"<T{thread_index}>".ljust(7)
        for i in range(rank + 1):
            index = int((rank * (rank + 1) / 2 + i) * partition_size + thread_index)
            indices.append(index)
            # string += f" {int(thread_index * max_index + (rank * (rank + 1) / 2 + i)):<5}" # row-wise
            string += f"{int((rank * (rank + 1) / 2 + i) * partition_size + thread_index):<6}" # column-wise
        print(string)



###############################################################################
correctness = False
if (len(indices) == partition_size * max_index and len(indices) == len(set(indices))):
    correctness = True
else:
    sorted_indices = sorted(indices)
    for i in range(len(indices)):
        if (i != sorted_indices[i]):
            print("Error:", i, "!=", sorted_indices[i])
print(f"Last index = {int(partition_size * max_index) - 1}, correct = {correctness}")


for rank in range(10):
    for subrank in range(rank + 1):
        print(int((rank * (rank + 1) / 2 + subrank)))