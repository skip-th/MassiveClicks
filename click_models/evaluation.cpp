/** Click model evaluation.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * evaluation.cpp:
 *  - Defines the functions for evaluating a CM.
 */

#include "evaluation.h"

/**
 * @brief Construct a new LogLikelihood object using a given click model.
 *
 * @param cm The click model for which the log-likelihood will be calculated.
 */
LogLikelihood::LogLikelihood(ClickModel_Hst *cm) {
    this->cm = cm;
}

/**
 * @brief Computes log-likelihood of the click model for a given set of testing
 * queries.
 *
 * @param testing_queries The set of testing queries.
 * @return The log-likelihood of the click model.
 */
float LogLikelihood::evaluate(TestSet& testing_queries) {
    // Go through all query sessions in the test set.
    for (SERP_Hst query_session : testing_queries) {
        std::vector<float> log_click_probs; //(MAX_SERP, 0);
        log_click_probs.reserve(MAX_SERP);

        // Get the log conditional click probability for each search result.
        this->cm->get_log_conditional_click_probs(query_session, log_click_probs);

        // Take the average of the log conditional click probabilities for this query session.
        this->llh_values.push_back(std::accumulate(log_click_probs.begin(), log_click_probs.end(), 0.f) / MAX_SERP);
    }

    // Sum all the log conditional click probabilities. The average will be taken later too using the size of the test task.
    return std::accumulate(llh_values.begin(), llh_values.end(), 0.f);
}

/**
 * @brief Compute the perplexity of the given click model for the given set of
 * testing queries.
 *
 * @param cm The click model for which the perplexity will be computed.
 * @param testing_queries The set of testing queries.
 */
void Perplexity::evaluate(ClickModel_Hst* cm, TestSet& testing_queries) {
    // Get the size of the test task.
    this->task_size = static_cast<float>(testing_queries.size());

    // Go through all sessions in the test set.
    for (SERP_Hst query_session : testing_queries) {
        std::vector<float> full_click_probs;
        full_click_probs.reserve(MAX_SERP);

        // Get the full click probability for each search result.
        cm->get_full_click_probs(query_session, full_click_probs);

        // Add the full click probability to each rank.
        for (int i{0}; i < MAX_SERP; i++) {
            this->task_rank_perplexities[i] += std::log2(full_click_probs[i]);
        }
    }
}

/**
 * @brief Changes the perplexity and task size to the given arguments.
 *
 * @param task_rank_perplexities The new perplexity values for rank 1 to 10.
 * @param task_size The new task size.
 */
void Perplexity::import(std::array<float, MAX_SERP>& task_rank_perplexities, float& task_size) {
    this->task_rank_perplexities = task_rank_perplexities;
    this->task_size = task_size;
}
