/** Click model evaluation.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * evaluation.h:
 *  - Defines the functions for evaluating a CM.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_EVALUATION
#define CLICK_MODEL_EVALUATION

// System include.
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

// User include.
#include "base.cuh"
#include "../utils/definitions.h"


class LogLikelihood {
private:
    std::vector<float> llh_values;
    ClickModel_Hst* cm;
public:
    explicit LogLikelihood(ClickModel_Hst* cm);
    float evaluate(std::vector<SERP_Hst>& testing_queries);
};


class Perplexity {
public:
    std::array<float, MAX_SERP> task_rank_perplexities{{0}};
    float task_size{0};
    Perplexity() = default;
    Perplexity(Perplexity const &ppl_obj);
    void evaluate(ClickModel_Hst* cm, std::vector<SERP_Hst>& testing_queries);
    void import(std::array<float, MAX_SERP>& task_rank_perplexities, float& task_size);
};

#endif //CLICK_MODEL_EVALUATION
