/** Click model factor.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * factor.cu:
 *  - Defines the functions for processing the phi formula of CCM and DBN.
 */

#include "factor.cuh"

/**
 * @brief Set the necessary arguments to compute phi for CCM.
 *
 * @param click_probs The current click probabilities for this SERP.
 * @param exam_probs The current examination probabilities for this SERP.
 * @param click The click on the current document.
 * @param last_click_rank The rank of the last clicked document in this SERP.
 * @param rank The rank of this document.
 * @param attr The attractiveness of this query-document pair.
 * @param tau_1 The first continuation parameter.
 * @param tau_2 The second continuation parameter.
 * @param tau_3 The third continuation parameter.
 */
HST DEV CCMFactor::CCMFactor(float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1], int click, int last_click_rank, int rank, float attr, float tau_1, float tau_2, float tau_3) {
    this->click_probs = click_probs;
    this->exam_probs = exam_probs;
    this->click = click;
    this->last_click_rank = last_click_rank;
    this->rank = rank;
    this->attr = attr;
    this->tau_1 = tau_1;
    this->tau_2 = tau_2;
    this->tau_3 = tau_3;
}

/**
 * @brief Compute the phi function for CCM.
 *
 * @param x The x input value for phi.
 * @param y The y input value for phi.
 * @param z The z input value for phi.
 */
HST DEV float CCMFactor::compute(int x, int y, int z) {
    float log_prob = 0.f;

    if (this->click == 0) { // Use tau 1 in case the document has not been clicked.
        if (y == 1) {
            return 0.f;
        }

        #ifdef __CUDA_ARCH__
            log_prob += __logf(1 - this->attr);
        #else
            log_prob += std::log(1 - this->attr);
        #endif

        if (x == 1) {
            #ifdef __CUDA_ARCH__
                if (z == 1) {
                    log_prob += __logf(this->tau_1);
                }
                else {
                    log_prob += __logf(1 - this->tau_1);
                }
            #else
                if (z == 1) {
                    log_prob += std::log(this->tau_1);
                }
                else {
                    log_prob += std::log(1 - this->tau_1);
                }
            #endif
        }
        else if (z == 1) {
            return 0.f;
        }
    }
    else { // Use tau 2 or 3 in case the document has been clicked.
        if (x == 0) {
            return 0.f;
        }

        #ifdef __CUDA_ARCH__
            log_prob += __logf(this->attr);

            if (y == 0) {
                log_prob += __logf(1 - this->attr);

                if (z == 1) {
                    log_prob += __logf(this->tau_2);
                }
                else {
                    log_prob += __logf(1 - this->tau_2);
                }
            }
            else {
                log_prob += __logf(this->attr);

                if (z == 1) {
                    log_prob += __logf(this->tau_3);
                }
                else {
                    log_prob += __logf(1 - this->tau_3);
                }
            }
        #else
            log_prob += std::log(this->attr);

            if (y == 0) {
                log_prob += std::log(1 - this->attr);

                if (z == 1) {
                    log_prob += std::log(this->tau_2);
                }
                else {
                    log_prob += std::log(1 - this->tau_2);
                }
            }
            else {
                log_prob += std::log(this->attr);

                if (z == 1) {
                    log_prob += std::log(this->tau_3);
                }
                else {
                    log_prob += std::log(1 - this->tau_3);
                }
            }
        #endif
    }

    if (z == 0) {
        if (this->last_click_rank >= this->rank + 1) {
            return 0.f;
        }
    }
    else if (this->rank + 1 < MAX_SERP) {
        #ifdef __CUDA_ARCH__
            for (int res_itr = 0; res_itr < MAX_SERP - this->rank - 1; res_itr++) {
                log_prob += __logf(this->click_probs[this->rank + 1][res_itr]);
            }
        #else
            for (int res_itr = 0; res_itr < MAX_SERP - this->rank - 1; res_itr++) {
                log_prob += std::log(this->click_probs[this->rank + 1][res_itr]);
            }
        #endif
    }

    #ifdef __CUDA_ARCH__
        if (x == 1) {
            log_prob += __logf(this->exam_probs[this->rank]);
        }
        else {
            log_prob += __logf(1 - this->exam_probs[this->rank]);
        }

        return __expf(log_prob);
    #else
        if (x == 1) {
            log_prob += std::log(this->exam_probs[this->rank]);
        }
        else {
            log_prob += std::log(1 - this->exam_probs[this->rank]);
        }

        return std::exp(log_prob);
    #endif
}

/**
 * @brief Set the necessary arguments to compute phi for DBN.
 *
 * @param click_probs The current click probabilities for this SERP_Hst.
 * @param exam_probs The current examination probabilities for this SERP_Hst.
 * @param click The click on the current document.
 * @param last_click_rank The rank of the last clicked document in this SERP_Hst.
 * @param rank The rank of this document.
 * @param attr The attractiveness of this query-document pair.
 * @param sat The satisfaction with this query-document pair.
 * @param gamma The continuation parameter.
 */
HST DEV DBNFactor::DBNFactor(float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1], int click,
                     int last_click_rank, int rank, float attr, float sat, float gamma) {
    this->click_probs = click_probs;
    this->exam_probs = exam_probs;
    this->click = click;
    this->last_click_rank = last_click_rank;
    this->rank = rank;
    this->attr = attr;
    this->gamma = gamma;
    this->sat = sat;
}

/**
 * @brief Compute the phi function for DBN.
 *
 * @param x The x input value for phi.
 * @param y The y input value for phi.
 * @param z The z input value for phi.
 */
HST DEV float DBNFactor::compute(int x, int y, int z) {
    float log_prob = 0.f;

    if (this->click == 0){
        if (y == 1){
            return 0.f;
        }

        #ifdef __CUDA_ARCH__
            log_prob += __logf(1 - this->attr);
        #else
            log_prob += std::log(1 - this->attr);
        #endif

        if (x == 1) {
            #ifdef __CUDA_ARCH__
                if (z == 1) {
                    log_prob += __logf(this->gamma);
                }
                else {
                    log_prob += __logf(1 - this->gamma);
                }
            #else
                if (z == 1) {
                    log_prob += std::log(this->gamma);
                }
                else {
                    log_prob += std::log(1 - this->gamma);
                }
            #endif
        }
        else if (z == 1) {
            return 0.f;
        }
    }
    else {
        if (x == 0) {
            return 0.f;
        }

        #ifdef __CUDA_ARCH__
            log_prob += __logf(this->attr);

            if (y == 0) {
                log_prob += __logf(1 - this->sat);
                if (z == 1) {
                    log_prob += __logf(this->gamma);
                }
                else {
                    log_prob += __logf(1 - this->gamma);
                }
            }
            else {
                if (z == 1) {
                    return 0.f;
                }

                log_prob += __logf(this->sat);
            }
        #else
            log_prob += std::log(this->attr);

            if (y == 0) {
                log_prob += std::log(1 - this->sat);
                if (z == 1) {
                    log_prob += std::log(this->gamma);
                }
                else {
                    log_prob += std::log(1 - this->gamma);
                }
            }
            else {
                if (z == 1) {
                    return 0.f;
                }

                log_prob += std::log(this->sat);
            }
        #endif
    }

    if (z == 0) {
        if (this->last_click_rank >= this->rank + 1) {
            return 0.f;
        }
    }
    else if (this->rank + 1 < MAX_SERP) {
        #ifdef __CUDA_ARCH__
            for (int res_itr = 0; res_itr < MAX_SERP - this->rank - 1; res_itr++) {
                log_prob += __logf(this->click_probs[this->rank + 1][res_itr]);
            }
        #else
            for (int res_itr = 0; res_itr < MAX_SERP - this->rank - 1; res_itr++) {
                log_prob += std::log(this->click_probs[this->rank + 1][res_itr]);
            }
        #endif
    }

    float exam_val = this->exam_probs[this->rank];

    #ifdef __CUDA_ARCH__
        if (x == 1) {
            log_prob += __logf(exam_val);
        }
        else {
            log_prob += __logf(1 - exam_val);
        }

        return __expf(log_prob);
    #else
        if (x == 1) {
            log_prob += std::log(exam_val);
        }
        else {
            log_prob += std::log(1 - exam_val);
        }

        return std::exp(log_prob);
    #endif
}