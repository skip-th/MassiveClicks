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
 * @param config Processing configuration.
 */
void check_valid_threads(ProcessingConfig& config) {
    if (config.thread_count < 1) {
        if (config.node_id == ROOT_RANK) {
            std::cerr << "[" << config.host_name << "] \033[12;31mError\033[0m: Invalid number of threads: " << config.thread_count << std::endl;
        }
        Communicate::finalize();
        exit(EXIT_SUCCESS);
    }
    else if (config.thread_count > static_cast<int>(std::thread::hardware_concurrency())) {
        std::cout << "[" << config.host_name << "] \033[12;33mWarning\033[0m: " << config.thread_count << " threads exceeds hardware concurrency of " << std::thread::hardware_concurrency() << " threads!" << std::endl;
    }
}

/**
 * @brief Check if the given number of devices is valid.
 *
 * @param config Processing configuration.
 */
void check_valid_devices(ProcessingConfig& config) {
    // Check if there are any GPU devices on this node to support the requested
    // GPU execution.
    if (config.exec_mode == 0 || config.exec_mode == 2) {
        if (config.device_count == 0 || config.max_gpus <= 0) { // No GPUs requested or found.
            Communicate::error_check("[" + std::string(config.host_name) + "] \033[12;31mError\033[0m: No GPU devices found for GPU-only or hybrid execution.");
        }
        else if (config.max_gpus > config.device_count) { // More GPUs requested than available.
            std::cerr << "[" << config.host_name << "] \033[12;33mWarning\033[0m: Number of GPUs requested (" << config.max_gpus << ") exceeds number of available devices (" << config.device_count << ")." << std::endl;
        }
    }
    Communicate::error_check();
}

/**
 * @brief Check if the given partitioning scheme is valid.
 *
 * @param config Processing configuration.
 */
void check_valid_partitioning(ProcessingConfig& config) {
    // Check if the provided partitioning scheme exists.
    if (config.partitioning_type < 0 || config.partitioning_type > 5) {
        if (config.node_id == ROOT_RANK) {
            std::cerr << "[" << config.host_name << "] \033[12;31mError\033[0m: Invalid partitioning scheme: " << config.partitioning_type << std::endl;
        }
        Communicate::finalize();
        exit(EXIT_SUCCESS);
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
    ProcessingConfig config;
    Communicate::initiate(argc, argv, config.total_nodes, config.node_id);

    // Get the properties of all nodes and their devices.
    ClusterProperties cluster_properties = \
        Communicate::gather_properties(get_node_properties(config.node_id));

    // Get number of devices on this node and their compute capability.
    config.device_count = cluster_properties.nodes[config.node_id].devices.size();
    int workers = config.device_count > 0 ? config.device_count : 1;

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
        {2, "Proportional Maximum Utilization"},
        {3, "Newest Architecture First"},
        {4, "Relative CUDA Cores"},
        {5, "Relative Peak Performance"},
    };

    std::map<int, std::string> execution_modes {
        {0, "GPU-only"},
        {1, "CPU-only"},
        {2, "Hybrid"},
    };

    // Create the configuration object.
    config.dataset_path      = "YandexRelPredChallenge100k.txt";
    config.output_path       = "";
    config.thread_count      = static_cast<int>(std::thread::hardware_concurrency());
    config.max_gpus          = cluster_properties.nodes[config.node_id].devices.size();
    config.iterations        = 50;
    config.max_sessions      = 40000;
    config.model_type        = 0;
    config.partitioning_type = 0;
    config.job_id            = 0;
    config.exec_mode         = cluster_properties.nodes[config.node_id].devices.size() > 0 ? 0 : 1; // GPU-only: 0, CPU-only: 1.
    config.test_share        = 0.2;
    config.help              = false;
    gethostname(config.host_name, HOST_NAME_MAX);

    // Define a struct to store the long and short flag forms, and handler for each option.
    struct Option {
        std::string long_form;
        std::string short_form;
        std::function<void(const char*)> handler;
    };

    // Define the available options as long and short form flags.
    std::vector<Option> options = {
        {"--n-threads",      "-n", [&](const char* value) { config.thread_count = std::stoi(value); }},
        {"--n-gpus",         "-g", [&](const char* value) { config.max_gpus = std::stoi(value); }},
        {"--raw-path",       "-r", [&](const char* value) { config.dataset_path = value; }},
        {"--output",         "-o", [&](const char* value) { config.output_path = value; }},
        {"--itr",            "-i", [&](const char* value) { config.iterations = std::stoi(value); }},
        {"--max-sessions",   "-s", [&](const char* value) { config.max_sessions = std::stoi(value); }},
        {"--model-type",     "-m", [&](const char* value) { config.model_type = std::stoi(value); }},
        {"--partition-type", "-p", [&](const char* value) { config.partitioning_type = std::stoi(value); }},
        {"--test-share",     "-t", [&](const char* value) { config.test_share = std::stod(value); }},
        {"--job-id",         "-j", [&](const char* value) { config.job_id = std::stoi(value); }},
        {"--exec-mode",      "-e", [&](const char* value) { config.exec_mode = std::stoi(value); }},
        {"--help",           "-h", [&](const char*)       { config.help = true; }},
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
                    if (config.node_id == ROOT_RANK) {
                        std::cerr << "[" << config.host_name << "] \033[12;31mError\033[0m: Did not recognize flag \'" << arg_name << "\'." << std::endl;
                    }
                    config.help = true;
                }
            }
            catch (std::invalid_argument& e) {
                if (config.node_id == ROOT_RANK) {
                    std::cerr << "[" << config.host_name << "] \033[12;31mError\033[0m: Invalid argument \'" << arg_value << "\' for flag \'" << arg_name << "\'." << std::endl;
                }
                config.help = true;
            }
        }
    }


    //-----------------------------------------------------------------------//
    // Error check the input                                                 //
    //-----------------------------------------------------------------------//

    // Check if help is requested.
    if (config.help) {
        if (config.node_id == ROOT_RANK) {
            show_help_msg();
        }
        Communicate::finalize();
        exit(EXIT_SUCCESS);
    }

    // Check if execution mode is valid
    if (config.node_id == ROOT_RANK && config.exec_mode == 2) {
        std::cerr << "[" << config.host_name << "] \033[12;33mWarning\033[0m: Hybrid execution is not yet supported. Defaulting to GPU-only." << std::endl;
    }

    // Check device, thread, and partitioning validity.
    check_valid_devices(config);
    check_valid_threads(config);
    check_valid_partitioning(config);

    // Set the number of GPU devices to 0 when CPU-only execution is requested.
    int workers_per_node[config.total_nodes];
    config.devices_per_node = workers_per_node;
    if (config.exec_mode == 1) {
        config.device_count = 0;
        config.max_gpus = 0;
        workers = 1;
        for (int nid = 0; nid < config.total_nodes; nid++) {
            workers_per_node[nid] = 1;
        }
    }
    // When GPU-only or hybrid execution is requested, ensure that cluster
    // properties follow the maximum number of GPUs.
    else if (config.exec_mode == 0 || config.exec_mode == 2) {
        for (int nid = 0; nid < config.total_nodes; nid++) {
            if (cluster_properties.nodes[nid].devices.size() > config.max_gpus) {
                cluster_properties.nodes[nid].devices.resize(config.max_gpus);
                cluster_properties.nodes[nid].host.device_count = config.max_gpus;
            }
            workers_per_node[nid] = cluster_properties.nodes[nid].devices.size();
        }
        config.device_count = std::min(config.device_count, config.max_gpus);
        workers = std::min(workers, config.device_count);
    }
    cluster_properties.nodes[config.node_id].host.thread_count = config.thread_count;


    //-----------------------------------------------------------------------//
    // Communicate system properties                                         //
    //-----------------------------------------------------------------------//

    // Compute the total number of usable workers in the cluster.
    config.worker_count = workers;
    int total_workers = (config.exec_mode == 0 || config.exec_mode == 2) ? std::accumulate(cluster_properties.nodes.begin(), cluster_properties.nodes.end(), 0,
             [](int sum, const NodeProperties& node) { return sum + node.devices.size(); }) : config.total_nodes;

    // Show job information on the root node.
    if (config.node_id == ROOT_RANK) {
        std::cout <<   "Job ID: "                         << config.job_id
                  << "\nNumber of machines: "             << config.total_nodes
                  << "\nNumber of devices in total: "     << total_workers
                  << "\nNumber of threads: "              << config.thread_count
                  << "\nExecution mode: "                 << execution_modes[config.exec_mode]
                  << "\nRaw data path: "                  << config.dataset_path
                  << "\nNumber of EM iterations: "        << config.iterations
                  << "\nShare of data used for testing: " << config.test_share * 100 << "%"
                  << "\nMax number of sessions: "         << config.max_sessions
                  << "\nPartitioning type: "              << partitioning_types[config.partitioning_type]
                  << "\nModel type: "                     << supported_click_models[config.model_type] << std::endl << std::endl;
        std::cout << "Node  Device  Arch  Free memory" << std::endl;
        for (int nid = 0; nid < config.total_nodes; nid++) {
            for (int did = 0; did < (config.exec_mode == 0 || config.exec_mode == 2 ? cluster_properties.nodes[nid].devices.size() : 1); did++) {
                int architecture = (config.exec_mode == 0 || config.exec_mode == 2 ? cluster_properties.nodes[nid].devices[did].compute_capability : -1);
                size_t memory = (config.exec_mode == 0 || config.exec_mode == 2 ? cluster_properties.nodes[nid].devices[did].available_memory : cluster_properties.nodes[nid].host.free_memory);
                std::cout << std::left << std::setw(5) << "N" + std::to_string(nid) << " "
                          << std::left << std::setw(7) << ((config.exec_mode == 0 || config.exec_mode == 2) ? "G" + std::to_string(did) : "C" + std::to_string(did)) << " "
                          << std::left << std::setw(5) << (architecture == -1 ? " " : std::to_string(architecture)) << " " << memory / 1e6 << std::endl;
            }
        }
        std::cout << std::endl;
    }


    //-----------------------------------------------------------------------//
    // Parse given click log dataset                                         //
    //-----------------------------------------------------------------------//

    timer.start("Preprocessing");

    LocalPartitions device_partitions(workers); // Device ID -> [train set, test set, size qd pairs]
    if (config.node_id == ROOT_RANK) { // Send partitions
        std::cout << "Parsing dataset..." << std::endl;

        if (parse_dataset(cluster_properties, config, device_partitions)) {
            Communicate::error_check("[" + std::string(config.host_name) + "] \033[12;31mError\033[0m: Unable to open the raw dataset.");
        }
    }
    else { // Receive partitions.
        if (parse_dataset(cluster_properties, config, device_partitions)) {
            Communicate::error_check("[" + std::string(config.host_name) + "] \033[12;31mError\033[0m: Parsing failed.");
        }
    }
    // Check if parsing failed on the root node.
    Communicate::error_check();

    timer.stop("Preprocessing");


    //-----------------------------------------------------------------------//
    // Run parallel generic EM algorithm on selected click model and dataset //
    //-----------------------------------------------------------------------//

    timer.start("EM");

    // Run click model parameter estimation using the generic EM algorithm.
    em_parallel(config, device_partitions);

    timer.stop("EM");


    //-----------------------------------------------------------------------//
    // Show metrics                                                          //
    //-----------------------------------------------------------------------//

    timer.stop("Total");

    auto print_time_metrics = [](const std::string& metric_name, double metric_value, double total_value) {
        auto percent = metric_value / total_value * 100;
        std::cout << std::left << std::setw(27) << metric_name << std::left << std::setw(7) << std::fixed << std::setprecision(7)
                  << metric_value << " seconds, " << std::right << std::setw(3) << std::setprecision(0) << std::round(percent) << " %" << std::endl;
    };

    // Show timing metrics on the root node.
    if (config.node_id == ROOT_RANK) {
        std::vector<double> percent_combined = { timer.elapsed("Preprocessing"), timer.elapsed("EM") };
        double combined_total = std::accumulate(percent_combined.begin(), percent_combined.end(), 0.0);
        std::cout << std::endl;
        print_time_metrics("Pre-processing time: ", timer.elapsed("Preprocessing"), combined_total);
        print_time_metrics("Parameter estimation time: ", timer.elapsed("EM"), combined_total);
        print_time_metrics("Total elapsed time: ", timer.elapsed("Total"), timer.elapsed("Total"));
        std::cout << std::endl;
    }

    // End MPI communication.
    Communicate::finalize();

    return EXIT_SUCCESS;
}
