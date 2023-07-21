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
#include <numeric>
#include <vector>

class Timer {
public:
    // Start a timer with a given label.
    void start(const std::string& label);

    // Stop a timer with a given label.
    void stop(const std::string& label);

    // Continue a timer with a given label.
    void cont(const std::string& label);

    // Record a lap with a given label and restart the timer. Returns the elapsed time of the previous lap.
    double lap(const std::string& label, bool restart = true);

    // Check the elapsed time of a timer with a given label.
    double elapsed(const std::string& label);

    // Get the average elapsed time over all stored laps of a specific label.
    double avg(const std::string& label);

    // Get the total elapsed time over all stored laps of a specific label.
    double total(const std::string& label);

    // Report elapsed time to stdout.
    void report(const std::string& label);
private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
    std::unordered_map<std::string, std::chrono::duration<double>> elapsed_times;
    std::unordered_map<std::string, std::vector<std::chrono::duration<double>>> laps;
};

#endif // TIMER_H