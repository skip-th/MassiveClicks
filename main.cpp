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
#include <iostream>
#include <map>
#include <string>
#include <cstddef>
#include <iomanip>
#include <cmath>
#include <thread>
#include <numeric>
#include <functional>

// User include.
#include "utils/definitions.h"
#include "utils/macros.cuh"
#include "utils/utils.cuh"
#include "utils/timer.h"
#include "parallel_em/parallel_em.cuh"
#include "parallel_em/communicator.h"
#include "data/dataset.h"
#include "click_models/base.cuh"

/**
 * @brief Check if the given number of threads is valid.
 *
 * @param n_threads Number of threads to use.
 * @param node_id MPI rank of this node.
 */
void check_valid_threads(int n_threads, int node_id) {
    if (n_threads < 1) {
        if (node_id == ROOT) {
            std::cerr << "[" << node_id << "] \033[12;31mError\033[0m: Invalid number of threads: " << n_threads << std::endl;
        }
        Communicate::finalize();
        exit(EXIT_SUCCESS);
    }
    else if (n_threads > static_cast<int>(std::thread::hardware_concurrency())) {
        std::cout << "[" << node_id << "] \033[12;33mWarning\033[0m: " << n_threads << " threads exceeds hardware concurrency of " << std::thread::hardware_concurrency() << " threads!" << std::endl;
    }
}

/**
 * @brief Check if the given number of devices is valid.
 *
 * @param n_devices Number of devices on this node.
 * @param n_gpus Number of GPUs to use.
 * @param exec_mode Execution mode.
 * @param node_id MPI rank of this node.
 */
void check_valid_devices(int n_devices, int n_gpus, int exec_mode, int node_id) {
    // Check if there are any GPU devices on this node to support the requested
    // GPU execution.
    if (exec_mode == 0 || exec_mode == 2) {
        if (n_devices == 0 || n_gpus <= 0) { // No GPUs requested or found.
            Communicate::error_check("[" + std::to_string(node_id) + "] \033[12;31mError\033[0m: No GPU devices found for GPU-only or hybrid execution.");
        }
        else if (n_gpus > n_devices) { // More GPUs requested than available.
            std::cerr << "[" << node_id << "] \033[12;33mWarning\033[0m: Number of GPUs requested (" << n_gpus << ") exceeds number of available devices (" << n_devices << ")." << std::endl;
        }
    }
    Communicate::error_check();
}


int main(int argc, char** argv) {
    Timer timer;
    timer.start("Total");

    //-----------------------------------------------------------------------//
    // Initialize MPI                                                        //
    //-----------------------------------------------------------------------//

    // Initialize MPI and get this node's rank and the number of nodes.
    int n_nodes, node_id;
    Communicate::initiate(argc, argv, n_nodes, node_id);

    // Get number of devices on this node and their compute capability.
    int n_devices{0};
    int n_devices_network[n_nodes];
    std::vector<std::vector<std::vector<int>>> network_properties(n_nodes); // Node, Device, [Architecture, Free memory].
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

    std::string raw_dataset_path = "YandexRelPredChallenge100k.txt";
    std::string output_path      = "";
    int n_threads                = static_cast<int>(std::thread::hardware_concurrency());
    int n_gpus                   = n_devices;
    int n_iterations             = 50;
    int max_sessions             = 40000;
    int model_type               = 0;
    int partitioning_type        = 0;
    int job_id                   = 0;
    int total_n_devices          = 0;
    int exec_mode                = n_devices > 0 ? 0 : 1; // GPU-only: 0, CPU-only: 1.
    float test_share             = 0.2;
    bool help                    = false;

    struct Option {
        std::string long_form;
        std::string short_form;
        std::function<void(const char*)> handler;
    };

    // Define the available options.
    std::vector<Option> options = {
        // {long_form, short_form, handler}
        {"--n-threads",      "-n", [&](const char* value) { n_threads = std::stoi(value); }},
        {"--n-gpus",         "-g", [&](const char* value) { n_gpus = std::stoi(value); }},
        {"--raw-path",       "-r", [&](const char* value) { raw_dataset_path = value; }},
        {"--output",         "-o", [&](const char* value) { output_path = value; }},
        {"--itr",            "-i", [&](const char* value) { n_iterations = std::stoi(value); }},
        {"--max-sessions",   "-s", [&](const char* value) { max_sessions = std::stoi(value); }},
        {"--model-type",     "-m", [&](const char* value) { model_type = std::stoi(value); }},
        {"--partition-type", "-p", [&](const char* value) { partitioning_type = std::stoi(value); }},
        {"--test-share",     "-t", [&](const char* value) { test_share = std::stod(value); }},
        {"--job-id",         "-j", [&](const char* value) { job_id = std::stoi(value); }},
        {"--exec-mode",      "-e", [&](const char* value) { exec_mode = std::stoi(value); }},
        {"--help",           "-h", [&](const char*)       { help = true; }},
    };

    // Parse input parameters.
    if (argc > 1) {
        for (int i = 1; i < argc; i += 2) {
            std::string arg_name = argv[i];
            const char* arg_value = argv[i + 1];

            bool handled = false;
            try {
                for (const auto& option : options) {
                    if (arg_name == option.long_form || arg_name == option.short_form) {
                        option.handler(arg_value);
                        handled = true;
                        break;
                    }
                }

                if (!handled && arg_name.rfind("-", 0) == 0) {
                    if (node_id == ROOT) {
                        std::cerr << "[" << node_id << "] \033[12;31mError\033[0m: Did not recognize flag \'" << arg_name << "\'." << std::endl;
                    }
                    help = true;
                }
            }
            catch (std::invalid_argument& e) {
                if (node_id == ROOT) {
                    std::cerr << "[" << node_id << "] \033[12;31mError\033[0m: Invalid argument \'" << arg_value << "\' for flag \'" << arg_name << "\'." << std::endl;
                }
                help = true;
            }
        }
    }

    //-----------------------------------------------------------------------//
    // Error check the input                                                 //
    //-----------------------------------------------------------------------//

    // Check if help is requested
    if (help) {
        if (node_id == ROOT) {
            show_help_msg();
        }
        Communicate::finalize();
        exit(EXIT_SUCCESS);
    }

    // Check if execution mode is valid
    if (node_id == ROOT && exec_mode == 2) {
        std::cerr << "[" << node_id << "] \033[12;33mWarning\033[0m: Hybrid execution is not yet supported. Defaulting to GPU-only." << std::endl;
    }

    // Check if devices are valid
    check_valid_devices(n_devices, n_gpus, exec_mode, node_id);

    // Check if number of threads is valid
    check_valid_threads(n_threads, node_id);

    // Set the number of GPU devices to 0 when CPU-only execution is requested.
    if (exec_mode == 1) {
        n_devices = 0;
        processing_units = 1;
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
    total_n_devices = std::accumulate(n_devices_network, n_devices_network + n_nodes, 0);

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

    timer.start("Preprocessing");
    timer.start("Parsing");

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

    timer.stop("Parsing");

    //-----------------------------------------------------------------------//
    // Partition parsed dataset                                              //
    //-----------------------------------------------------------------------//

    timer.start("Partitioning");

    if (node_id == ROOT) {
        // Split the dataset into partitions. One for each device on each node.
        dataset.make_splits(network_properties, test_share, partitioning_type, model_type);
    }

    timer.stop("Partitioning");

    //-----------------------------------------------------------------------//
    // Send/Retrieve partitions                                              //
    //-----------------------------------------------------------------------//

    timer.start("Communication");

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

    timer.stop("Communication");

    //-----------------------------------------------------------------------//
    // Sort dataset partitions                                               //
    //-----------------------------------------------------------------------//

    timer.start("Sorting");

    // Sorting the dataset is only necessary when the CPU is used.
    if (exec_mode == 1 || exec_mode == 2) {
        // Sort the training sets for each device by query so a multiple sessions
        // with the same query can be assigned to a single cpu thread.
        if (node_id == ROOT) {
            std::cout << "\nSorting dataset partitions..." << std::endl;
        }
        sort_partitions(device_partitions, n_threads);
    }

    timer.stop("Sorting");
    timer.stop("Preprocessing");

    //-----------------------------------------------------------------------//
    // Run parallel generic EM algorithm on selected click model and dataset //
    //-----------------------------------------------------------------------//

    timer.start("EM");

    // Run click model parameter estimation using the generic EM algorithm.
    em_parallel(model_type, node_id, n_nodes, n_threads, n_devices_network, n_iterations,
                exec_mode, n_devices, processing_units, device_partitions, output_path);

    timer.stop("EM");

    //-----------------------------------------------------------------------//
    // Show metrics                                                          //
    //-----------------------------------------------------------------------//

    timer.stop("Total");

    auto print_time_metrics = [](const std::string& metric_name, double metric_value, double total_value) {
        auto percent = metric_value / total_value * 100;
        std::cout << std::left << std::setw(27) << metric_name << std::left << std::setw(7) << std::fixed << std::setprecision(7) <<
            metric_value << " seconds, " << std::right << std::setw(3) << std::setprecision(0) << std::round(percent) << " %" << std::endl;
    };

    // Show timing metrics on the root node.
    if (node_id == ROOT) {
        std::vector<double> percent_preproc = { timer.elapsed("Parsing"), timer.elapsed("Partitioning"), timer.elapsed("Communication"), timer.elapsed("Sorting") };
        std::vector<double> percent_combined = { timer.elapsed("Preprocessing"), timer.elapsed("EM") };
        double preproc_total = std::accumulate(percent_preproc.begin(), percent_preproc.end(), 0.0);
        double combined_total = std::accumulate(percent_combined.begin(), percent_combined.end(), 0.0);

        std::cout << std::endl;
        print_time_metrics("Total pre-processing time: ", timer.elapsed("Preprocessing"), combined_total);
        print_time_metrics("  Parsing time: ", timer.elapsed("Parsing"), preproc_total);
        print_time_metrics("  Partitioning time: ", timer.elapsed("Partitioning"), preproc_total);
        print_time_metrics("  Communication time: ", timer.elapsed("Communication"), preproc_total);
        if (exec_mode == 1 || exec_mode == 2) {
            print_time_metrics("  Sorting time: ", timer.elapsed("Sorting"), preproc_total);
        }
        print_time_metrics("Parameter estimation time: ", timer.elapsed("EM"), combined_total);
        print_time_metrics("Total elapsed time: ", timer.elapsed("Total"), timer.elapsed("Total"));
        std::cout << std::endl;
    }

    // End MPI communication.
    Communicate::finalize();

    return EXIT_SUCCESS;
}
