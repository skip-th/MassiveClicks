# MassiveClicks: A Massively-parallel Framework for Efficient Click Models Training

MassiveClicks is a multi-node multi-GPU framework for training click models
using expectation maximization (EM). The framework supports heterogeneous GPU
architectures, variable numbers of GPUs per node, and allows for multi-node
multi-core CPU-based training when no GPUs are available. The following click
models are currently supported:

1. *Position-based Model (PBM)*.
2. *User Browsing Model (UBM)*.
3. *Click Chain Model (CCM)*.
4. *Dynamic Bayesian Network Model (DBN)*.

MassiveClicks builds upon the generic EM-based algorithm for CPU-based
single-node click model training, [ParClick](https://github.com/uva-sne/ParClick).

## Requirements

* CUDA version: 12.1
* MPI version: 3.1
* C++ version: C++11 or higher

## Installation

1. Clone the repository: `git clone https://github.com/skip-th/MassiveClicks.git`
2. Navigate to the project directory: `cd MassiveClicks`
3. Run the installation commands: `cmake . && make`

## Usage

Here is a basic example of how to run MassiveClicks:

`./mclicks --raw-path 'dataset.txt' --max-sessions 40000 --itr 50 --model-type 0 --partition-type 0 --test-share 0.2`

The above command will train a PBM click model on 40000 sessions from the
dataset `dataset.txt` for 50 iterations. Sessions are assigned to all available
GPUs in a round-robin fashion. 20% of the dataset is used as the test set.

The dataset queries are assumed to be in the following format:

```markdown
<session_id> <time_passed> <event_type> <query_id> <region_id> <document_id_0> ... <document_id_9>
```

And the clicks that follow a query are assumed to be in the following format:

```markdown
<session_id> <time_passed> <event_type> <document_id>
```

Only clicks containing a document ID occurring in the immediately preceding
query are considered.
