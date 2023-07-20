/** Timer header for timing code.
 *
 * timer.h:
 *  - Timer header implementation.
 */

#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>

class Timer {
public:
    // Start a timer with a given label.
    void start(const std::string& label);

    // Stop a timer with a given label.
    void stop(const std::string& label);

    // Check the elapsed time of a timer with a given label.
    double elapsed(const std::string& label);

    // Report elapsed time to stdout.
    void report(const std::string& label);
private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    std::unordered_map<std::string, std::chrono::duration<double>> elapsed_times;
};

#endif // TIMER_H