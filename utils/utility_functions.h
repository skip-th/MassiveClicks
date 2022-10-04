/** Several utility functions.
 *
 * utility_functions.h:
 *  - Defines several utility functions.
 */

#ifndef CLICK_MODEL_UTILS_H
#define CLICK_MODEL_UTILS_H

// System include.
#include <cstdlib>
#include <cmath>
#include <limits>
#include <iostream>

namespace Utils
{
    template <typename T>
    T sum(T *data, int size) {
        T sum = 0;

        for (int i = 0; i < size; i++) {
            sum += data[i];
        }

        return sum;
    }

    template <typename T>
    T max(T *data, int size) {
        T maximum = -INFINITY;

        for (int i = 0; i < size; i++) {
            if (data[i] > maximum) {
                maximum = data[i];
            }
        }

        return sum;
    }

    template <typename T>
    void percent_dist(T *src, double *dst, int size) {
        double total = sum(src, size);

        for (int i = 0; i < size; i++) {
            dst[i] = ((double) src[i]) / total;
        }
    }

    template <typename T>
    std::string digit_len(T digit, int size) {
        std::string digit_str = std::to_string(digit);

        if (size < digit_str.length()) {
            return digit_str.substr(0, size);
        }

        return digit_str;
    }
}

// Display the help message.
void print_help_msg(void);

#endif // CLICK_MODEL_UTILS_H