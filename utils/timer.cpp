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
        auto elapsed = stop_time - start_times[label];
        if (elapsed_times.find(label) != elapsed_times.end()) {
            elapsed_times[label] += elapsed;
        } else {
            elapsed_times[label] = elapsed;
        }
    } else {
        throw std::invalid_argument("Timer with label " + label + " was not started");
    }
}

void Timer::cont(const std::string& label) {
    if (elapsed_times.find(label) != elapsed_times.end()) {
        start_times[label] = std::chrono::high_resolution_clock::now();
    } else {
        throw std::invalid_argument("Timer with label " + label + " was not stopped or does not exist");
    }
}

double Timer::lap(const std::string& label, bool restart) {
    stop(label);
    if (elapsed_times.find(label) != elapsed_times.end()) {
        laps[label].push_back(elapsed_times[label]);
        if (restart) {
            start(label); // Restart the timer
        }
        return elapsed(label);
    } else {
        throw std::invalid_argument("Timer with label " + label + " was not stopped or does not exist");
    }
}

double Timer::elapsed(const std::string& label) {
    if (elapsed_times.find(label) != elapsed_times.end()) {
        return elapsed_times[label].count();
    } else {
        throw std::invalid_argument("Timer with label " + label + " was not stopped or does not exist");
    }
}

double Timer::avg(const std::string& label) {
    if (laps.find(label) != laps.end() && !laps[label].empty()) {
        std::chrono::duration<double> sum = std::accumulate(laps[label].begin(), laps[label].end(), std::chrono::duration<double>(0));
        return sum.count() / laps[label].size();
    } else {
        throw std::invalid_argument("No laps recorded for label " + label);
    }
}

double Timer::total(const std::string& label) {
    if (laps.find(label) != laps.end() && !laps[label].empty()) {
        std::chrono::duration<double> sum = std::accumulate(laps[label].begin(), laps[label].end(), std::chrono::duration<double>(0));
        return sum.count();
    } else {
        throw std::invalid_argument("No laps recorded for label " + label);
    }
}

void Timer::report(const std::string& label) {
    std::cout << label << ": " << elapsed(label) << "s" << std::endl;
}
