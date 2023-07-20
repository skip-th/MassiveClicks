/** Timer class for timing code.
 *
 * timer.cpp:
 *  - Timer class implementation.
 */

#include "timer.h"

void Timer::start(const std::string& label) {
    start_times[label] = std::chrono::high_resolution_clock::now();
}

void Timer::stop(const std::string& label) {
    auto stop_time = std::chrono::high_resolution_clock::now();
    if (start_times.find(label) != start_times.end()) {
        elapsed_times[label] = stop_time - start_times[label];
    } else {
        throw std::invalid_argument("Timer with label " + label + " was not started");
    }
}

double Timer::elapsed(const std::string& label) {
    if (elapsed_times.find(label) != elapsed_times.end()) {
        return elapsed_times[label].count();
    } else {
        throw std::invalid_argument("Timer with label " + label + " was not stopped or does not exist");
    }
}

void Timer::report(const std::string& label) {
    std::cout << label << ": " << elapsed(label) << "s" << std::endl;
}