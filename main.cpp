/**
 * Multi-GPU training of EM-based Click Models.
 * Currently supported click models are PBM, UBM, CCM, and DBN.
 *
 * main.cpp:
 *  - Source file.
 *  - Initialize MPI.
 *  - Parse dataset.
 *  - Train EM-based click model.
 */

// MPI include.
#include <mpi.h>

// System include.
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <cstddef>
#include <iomanip>
#include <cmath>
#include <thread>
#include <numeric>

// User include.
#include "utils/definitions.h"
#include "utils/macros.cuh"
#include "utils/utils.cuh"
#include "parallel_em/parallel_em.cuh"
#include "parallel_em/communicator.h"
#include "data/dataset.h"
#include "click_models/base.cuh"


int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();

    //-----------------------------------------------------------------------//
    // Initialize MPI                                                        //
    //-----------------------------------------------------------------------//

    // Initialize MPI and get this node's rank and the number of nodes.
    int n_nodes, node_id;
    Communicate::initiate(argc, argv, n_nodes, node_id);

    // Get number of GPU devices available on this node.
    int n_devices{0};
    int n_devices_network[n_nodes];
    std::vector<std::vector<std::vector<int>>> network_properties(n_nodes); // Node, Device, [Architecture, Free memory].

    // Get number of devices on this node and their compute capability.
    get_number_devices(&n_devices);
    int processing_units = n_devices > 0 ? n_devices : 1;


    //-----------------------------------------------------------------------//
    // Declare and retrieve input parameters                                 //
    //-----------------------------------------------------------------------//

    std::map<int, std::string> supported_click_models {
        {0, "PBM"},
        {1, "CCM"},
        {2, "DBN"},
        {3, "UBM"}
    };

    std::map<int, std::string> partitioning_types {
        {0, "Round-Robin"},
        {1, "Maximum Utilization"},
        {2, "Resource-Aware Maximum Utilization"},
        {3, "Newest architecture first"},
    };

    std::map<int, std::string> execution_modes {
        {0, "GPU-only"},
        {1, "CPU-only"},
        {2, "Hybrid"},
    };

    std::string raw_dataset_path{"YandexRelPredChallenge100k.txt"};
    std::string output_path{""};
    int n_threads{static_cast<int>(std::thread::hardware_concurrency())};
    int n_gpus{n_devices};
    int n_iterations{50};
    int max_sessions{40000};
    int model_type{0};
    int partitioning_type{0};
    int job_id{0};
    float test_share{0.2};
    int total_n_devices{0};
    int exec_mode{n_devices > 0 ? 0 : 1}; // GPU-only: 0, CPU-only: 1.
    bool help{false};

    // Parse input parameters.
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            try {
                if (std::string(argv[i]) == "--n-threads" || std::string(argv[i]) == "-n") {
                    n_threads = std::stoi(argv[i+1]);
                }
                else if (std::string(argv[i]) == "--n-gpus" || std::string(argv[i]) == "-g") {
                    n_gpus = std::stoi(argv[i+1]);
                }
                else if (std::string(argv[i]) == "--raw-path" || std::string(argv[i]) == "-r") {
                    raw_dataset_path = argv[i + 1];
                }
                else if (std::string(argv[i]) == "--output" || std::string(argv[i]) == "-o") {
                    output_path = argv[i + 1];
                }
                else if (std::string(argv[i]) == "--itr" || std::string(argv[i]) == "-i") {
                    n_iterations = std::stoi(argv[i + 1]);
                }
                else if (std::string(argv[i]) == "--max-sessions" || std::string(argv[i]) == "-s") {
                    max_sessions = std::stoi(argv[i + 1]);
                }
                else if (std::string(argv[i]) == "--model-type" || std::string(argv[i]) == "-m") {
                    model_type = std::stoi(argv[i + 1]);
                }
                else if (std::string(argv[i]) == "--partition-type" || std::string(argv[i]) == "-p") {
                    partitioning_type = std::stoi(argv[i + 1]);
                }
                else if (std::string(argv[i]) == "--test-share" || std::string(argv[i]) == "-t") {
                    test_share = std::stod(argv[i + 1]);
                }
                else if (std::string(argv[i]) == "--job-id" || std::string(argv[i]) == "-j") {
                    job_id = std::stoi(argv[i + 1]);
                }
                else if (std::string(argv[i]) == "--exec-mode" || std::string(argv[i]) == "-e") {
                    exec_mode = std::stoi(argv[i + 1]);
                }
                else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
                    help = true;
                }
                else if (std::string(argv[i]).rfind("-", 0) == 0) {
                    if (node_id == ROOT) {
                        std::cout << "Did not recognize flag \'" <<
                        std::string(argv[i]) << "\'." << std::endl;
                    }
                    help = true;
                }
            }
            catch (std::invalid_argument& e) {
                if (node_id == ROOT) {
                    std::cout << "Invalid argument \'" << argv[i + 1] <<
                    "\' for flag \'" << argv[i] << "\'." << std::endl;
                }
                help = true;
            }
        }
    }


    //-----------------------------------------------------------------------//
    // Error check the input                                                 //
    //-----------------------------------------------------------------------//

    // Display help message and shutdown.
    if (help) {
        if (node_id == ROOT) {
            show_help_msg();
        }

        // End MPI communication and exit.
        Communicate::finalize();
        return EXIT_SUCCESS;
    }

    if (node_id == ROOT && exec_mode == 2) {
        std::cerr << "[" << node_id << "] \033[12;33mWarning\033[0m: Hybrid execution is not yet supported. Defaulting to GPU-only." << std::endl;
    }

    // Ensure that the execution mode can be used.
    if (exec_mode == 0 || exec_mode == 2) {
        if (n_devices == 0 || n_gpus <= 0) {
            Communicate::error_check("[" + std::to_string(node_id) + "] \033[12;31mError\033[0m: No GPU devices found for GPU-only or hybrid execution.");
        }
        else if (n_gpus > n_devices) {
            std::cerr << "[" << node_id << "] \033[12;33mWarning\033[0m: Number of GPUs requested (" << n_gpus << ") exceeds number of available devices (" << n_devices << ")." << std::endl;
        }
    }
    else if (exec_mode == 1) {
        n_devices = 0;
        processing_units = 1;
    }
    Communicate::error_check();

    // Ensure that the number of threads is valid.
    if (n_threads < 1) {
        if (node_id == ROOT) {
            std::cerr << "[" << node_id << "] \033[12;31mError\033[0m: Invalid number of threads: " << n_threads << std::endl;
        }

        // End MPI communication and exit.
        Communicate::finalize();
        return EXIT_SUCCESS;
    }
    else if (n_threads > static_cast<int>(std::thread::hardware_concurrency())) {
        std::cout << "[" << node_id << "] \033[12;33mWarning\033[0m: " << n_threads << " threads exceeds hardware concurrency of " << std::thread::hardware_concurrency() << " threads!" << std::endl;
    }


    //-----------------------------------------------------------------------//
    // Communicate system properties                                         //
    //-----------------------------------------------------------------------//

    // Communicate the number of devices to the root node.
    if (exec_mode == 0 || exec_mode == 2) {
        n_devices = std::min(n_devices, n_gpus);
        processing_units = std::min(processing_units, n_gpus);
        Communicate::get_n_devices(processing_units, n_devices_network);
    }
    else if (exec_mode == 1) {
        Communicate::get_n_devices(1, n_devices_network);
    }

    // Get the compute architecture and free memory of each device on this node.
    int device_architecture[processing_units], free_memory[processing_units];
    if (exec_mode == 0 || exec_mode == 2) {
        for (int device_id = 0; device_id < n_devices; device_id++) {
            device_architecture[device_id] = get_compute_capability(device_id);

            size_t fmem, tmem;
            get_device_memory(device_id, fmem, tmem, 1e6);
            free_memory[device_id] = fmem;
        }
    }
    else {
        // Retrieve the available system memory instead of GPU memory.
        device_architecture[0] = -1;

        size_t fmem, tmem;
        get_host_memory(fmem, tmem, 1e6);
        free_memory[0] = fmem;
    }

    // Gather the compute architectures and free memory on the root node.
    Communicate::gather_properties(node_id, n_nodes, processing_units, n_devices_network, network_properties, device_architecture, free_memory);
    total_n_devices = std::accumulate(n_devices_network, n_devices_network+n_nodes, 0);

    // Show job information on the root node.
    if (node_id == ROOT) {
        std::cout << "Job ID: " << job_id <<
        "\nNumber of machines: " << n_nodes <<
        "\nNumber of devices in total: " << total_n_devices <<
        "\nNumber of threads: " << n_threads <<
        "\nExecution mode: " << execution_modes[exec_mode] <<
        "\nRaw data path: " << raw_dataset_path <<
        "\nNumber of EM iterations: " << n_iterations <<
        "\nShare of data used for testing: " << test_share * 100 << "%" <<
        "\nMax number of sessions: " << max_sessions <<
        "\nPartitioning type: " << partitioning_types[partitioning_type] <<
        "\nModel type: " << supported_click_models[model_type] << std::endl << std::endl;

        std::cout << "Node  Device  Arch  Free memory" << std::endl;
        for (int nid = 0; nid < n_nodes; nid++) {
            for (int did = 0; did < n_devices_network[nid]; did++) {
                int architecture = network_properties[nid][did][0];
                std::cout << std::left << std::setw(5) << "N" + std::to_string(nid) << " " <<
                std::left << std::setw(7) << ((exec_mode == 0 || exec_mode == 2) ? "G" + std::to_string(did) : "C" + std::to_string(did)) << " " <<
                std::left << std::setw(5) << (architecture == -1 ? " " : std::to_string(architecture)) << " " <<
                network_properties[nid][did][1] << std::endl;
            }
        }
        std::cout << std::endl;
    }


    //-----------------------------------------------------------------------//
    // Parse given click log dataset                                         //
    //-----------------------------------------------------------------------//

    auto preprocessing_start_time = std::chrono::high_resolution_clock::now();
    auto parse_start_time = std::chrono::high_resolution_clock::now();

    Dataset dataset;

    if (node_id == ROOT) {
        std::cout << "Parsing dataset." << std::endl;

        if (parse_dataset(dataset, raw_dataset_path, max_sessions)) {
            Communicate::error_check("[" + std::to_string(node_id) + "] \033[12;31mError\033[0m: Unable to open the raw dataset.");
        }

        std::cout << "Found " << dataset.size_queries() << " query sessions." << std::endl;
    }
    // Check if parsing failed on the root node.
    Communicate::error_check();

    auto parse_stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_parsing = parse_stop_time - parse_start_time;

    //-----------------------------------------------------------------------//
    // Partition parsed dataset                                              //
    //-----------------------------------------------------------------------//

    auto partitioning_start_time = std::chrono::high_resolution_clock::now();

    if (node_id == ROOT) {
        // Split the dataset into partitions. One for each device on each node.
        dataset.make_splits(network_properties, test_share, partitioning_type, model_type);
    }

    auto partitioning_stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_partitioning = partitioning_stop_time - partitioning_start_time;

    //-----------------------------------------------------------------------//
    // Send/Retrieve partitions                                              //
    //-----------------------------------------------------------------------//

    auto transfering_start_time = std::chrono::high_resolution_clock::now();

    // Store the train/test splits for each device on each node.
    std::vector<std::tuple<std::vector<SERP_Hst>, std::vector<SERP_Hst>, int>> device_partitions(processing_units); // Device ID -> [train set, test set, size qd pairs]

    // Communicate the training sets for each device to their node.
    Communicate::send_partitions(node_id, n_nodes, processing_units, total_n_devices, n_devices_network, dataset, device_partitions);

    // Show information about the distributed partitions on the root node.
    if (node_id == ROOT) {
        std::cout << "\nNode  Device  Train queries  Test queries  QD-pairs" << std::endl;
        for (int nid = 0; nid < n_nodes; nid++) {
            for (int did = 0; did < n_devices_network[nid]; did++) {
                std::cout << std::left << std::setw(5) << "N" + std::to_string(nid) << " " <<
                std::left << std::setw(7) << ((exec_mode == 0 || exec_mode == 2) ? "G" + std::to_string(did) : "C" + std::to_string(did)) << " " <<
                std::left << std::setw(14) << (nid == ROOT ? std::get<0>(device_partitions[did]).size() : dataset.size_train(nid, did)) << " " <<
                std::left << std::setw(13) << (nid == ROOT ? std::get<1>(device_partitions[did]).size() : dataset.size_test(nid, did)) << " " <<
                (nid == ROOT ? std::get<2>(device_partitions[did]) : dataset.size_qd(nid, did)) << std::endl;
            }
        }
    }

    // Wait until printing is done.
    Communicate::barrier();

    auto transfering_stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_transfering = transfering_stop_time - transfering_start_time;


    //-----------------------------------------------------------------------//
    // Sort dataset partitions                                               //
    //-----------------------------------------------------------------------//

    auto sorting_start_time = std::chrono::high_resolution_clock::now();

    // Sorting the dataset is only necessary when the CPU is used.
    if (exec_mode == 1 || exec_mode == 2) {
        // Sort the training sets for each device by query so a multiple sessions
        // with the same query can be assigned to a single cpu thread.
        if (node_id == ROOT) {
            std::cout << "\nSorting dataset partitions..." << std::endl;
        }
        sort_partitions(device_partitions, n_threads);
    }

    auto sorting_stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_sorting = sorting_stop_time - sorting_start_time;

    auto preprocessing_stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_preprocessing = preprocessing_stop_time - preprocessing_start_time;


    //-----------------------------------------------------------------------//
    // Run parallel generic EM algorithm on selected click model and dataset //
    //-----------------------------------------------------------------------//

    auto estimating_start_time = std::chrono::high_resolution_clock::now();

    // Run click model parameter estimation using the generic EM algorithm.
    em_parallel(model_type, node_id, n_nodes, n_threads, n_devices_network, n_iterations,
                exec_mode, n_devices, processing_units, device_partitions, output_path);

    auto estimating_stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_estimating = estimating_stop_time - estimating_start_time;


    //-----------------------------------------------------------------------//
    // Show metrics                                                          //
    //-----------------------------------------------------------------------//

    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop_time - start_time;

    // Show timing metrics on the root node.
    if (node_id == ROOT) {
        std::vector<double> percent_preproc = { elapsed_parsing.count(), elapsed_partitioning.count(), elapsed_transfering.count(), elapsed_sorting.count() };
        std::vector<double> percent_combined = { elapsed_preprocessing.count(), elapsed_estimating.count() };
        double preproc_total = std::accumulate(percent_preproc.begin(), percent_preproc.end(), 0.0);
        double combined_total = std::accumulate(percent_combined.begin(), percent_combined.end(), 0.0);
        for (size_t i = 0; i < percent_preproc.size(); i++) { percent_preproc[i] = (percent_preproc[i] / preproc_total); }
        for (size_t i = 0; i < percent_combined.size(); i++) { percent_combined[i] = (percent_combined[i] / combined_total); }
        auto digit_cutoff = [](double digit, int size) { return std::to_string(digit).substr(0, std::to_string(digit).find(".") + size + 1); };

        std::cout << std::endl << std::left << std::setw(27) << "Total pre-processing time: " << std::left << std::setw(7) << digit_cutoff(elapsed_preprocessing.count(), 7) << " seconds, " << std::right << std::setw(3) << std::round(percent_combined[0] * 100) << " %" << std::endl;
        std::cout << std::left << std::setw(27) << "  Parsing time: " << std::left << std::setw(7) << digit_cutoff(elapsed_parsing.count(), 7) << " seconds, " << std::right << std::setw(3) << std::round(percent_preproc[0] * 100) << " %" << std::endl;
        std::cout << std::left << std::setw(27) << "  Partitioning time: " << std::left << std::setw(7) << digit_cutoff(elapsed_partitioning.count(), 7) << " seconds, " << std::right << std::setw(3) << std::round(percent_preproc[1] * 100) << " %" << std::endl;
        std::cout << std::left << std::setw(27) << "  Communication time: " << std::left << std::setw(7) << digit_cutoff(elapsed_transfering.count(), 7) << " seconds, " << std::right << std::setw(3) << std::round(percent_preproc[2] * 100) << " %" << std::endl;
        if (exec_mode == 1 || exec_mode == 2) {
            std::cout << std::left << std::setw(27) << "  Sorting time: " << std::left << std::setw(7) << digit_cutoff(elapsed_sorting.count(), 7) << " seconds, " << std::right << std::setw(3) << std::round(percent_preproc[3] * 100) << " %" << std::endl;
        }
        std::cout << std::left << std::setw(27) << "Parameter estimation time: " << std::left << std::setw(7) << digit_cutoff(elapsed_estimating.count(), 7) << " seconds, " << std::right << std::setw(3) << std::round(percent_combined[1] * 100) << " %" << std::endl;
        std::cout << std::left << std::setw(27) << "Total elapsed time: " << std::left << std::setw(7) << digit_cutoff(elapsed.count(), 7) << " seconds, 100 %" << std::endl << std::endl;
    }

    // End MPI communication.
    Communicate::finalize();

    return EXIT_SUCCESS;
}
