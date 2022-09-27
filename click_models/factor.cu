//
// Created by Pooya Khandel on 09/09/2021.
//

#include "factor.cuh"

/**
 * @brief Set the necessary arguments to compute phi.
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
DEV CCMFactor::CCMFactor(float (&click_probs)[MAX_SERP_LENGTH][MAX_SERP_LENGTH], float (&exam_probs)[MAX_SERP_LENGTH + 1], int click, int last_click_rank, int rank, float attr, float tau_1, float tau_2, float tau_3) {
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
 * @brief Compute the phi function.
 *
 * @param x The x input value for phi.
 * @param y The y input value for phi.
 * @param z The z input value for phi.
 */
DEV float CCMFactor::compute(int x, int y, int z, int qid, int did) {
    float log_prob = 0.f;
    // printf("%d, %d] (%d,%d,%d) [attr = %f, tau 1 = %f, tau 2 = %f, tau 3 = %f, last_click_rank = %d, rank = %d, click = %d\n", qid, did, x, y, z, this->attr, this->tau_1, this->tau_2, this->tau_3, this->last_click_rank, this->rank, this->click);
    // if (x == 1 && y == 0 && z == 1) {
    // }
    // int debug = 0;
    // if (x == 1 && y == 0 && z == 1) {
    //     debug = 1;
    // }
    // if (debug) {
        // for (int i = 0; i < MAX_SERP_LENGTH; i++) {
        //     for (int j = 0; j < MAX_SERP_LENGTH; j++) {
        //         printf("click_probs[%d][%d] = %f\n", i, j, this->click_probs[i][j]);
        //     }
        // }
        // for (int j = 0; j < MAX_SERP_LENGTH + 1; j++) {
        //     printf("exam_probs[%d] = %f\n", j, this->click_probs[j]);
        // }
    // }
    // if (debug) {printf("click = %d, last_click_rank = %d,  rank = %d, attr = %f, tau_1 = %f, tau_2 = %f, tau_3 = %f\n", this->click, this->last_click_rank, this->rank, this->attr, this->tau_1, this->tau_2, this->tau_3);}

    if (this->click == 0) { // Use tau 1 in case the document has not been clicked.
        if (y == 1) {
            // printf("%d, %d] (%d,%d,%d) 1, log_prob = %f, returned = %f, [attr = %f, tau 1 = %f, tau 2 = %f, tau 3 = %f, last_click_rank = %d, rank = %d, click = %d\n", qid, did, x, y, z, log_prob, 0.f, this->attr, this->tau_1, this->tau_2, this->tau_3, this->last_click_rank, this->rank, this->click);
            return 0.f;
        }

        log_prob += __logf(1 - this->attr);

        if (x == 1) {
            if (z == 1) {
                log_prob += __logf(this->tau_1);
                // if (debug) {printf("nc, tau1 = %f\n", this->tau_1);}
            }
            else {
                log_prob += __logf(1 - this->tau_1);
            }
        }
        else if (z == 1) {
            // printf("%d, %d] (%d,%d,%d) 2, log_prob = %f, returned = %f, [attr = %f, tau 1 = %f, tau 2 = %f, tau 3 = %f, last_click_rank = %d, rank = %d, click = %d\n", qid, did, x, y, z, log_prob, 0.f, this->attr, this->tau_1, this->tau_2, this->tau_3, this->last_click_rank, this->rank, this->click);
            return 0.f;
        }
    }
    else { // Use tau 2 or 3 in case the document has been clicked.
        if (x == 0) {
            // printf("%d, %d] (%d,%d,%d) 3, log_prob = %f, returned = %f, [attr = %f, tau 1 = %f, tau 2 = %f, tau 3 = %f, last_click_rank = %d, rank = %d, click = %d\n", qid, did, x, y, z, log_prob, 0.f, this->attr, this->tau_1, this->tau_2, this->tau_3, this->last_click_rank, this->rank, this->click);
            return 0.f;
        }

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
    }

    if (z == 0) {
        if (this->last_click_rank >= (this->rank + 1)) {
            // printf("%d, %d] (%d,%d,%d) 4, log_prob = %f, returned = %f, [attr = %f, tau 1 = %f, tau 2 = %f, tau 3 = %f, last_click_rank = %d, rank = %d, click = %d\n", qid, did, x, y, z, log_prob, 0.f, this->attr, this->tau_1, this->tau_2, this->tau_3, this->last_click_rank, this->rank, this->click);
            return 0.f;
        }
    }
    else if (this->rank + 1 < MAX_SERP_LENGTH) {
        // for (int start_rank = this->rank + 1; start_rank < MAX_SERP_LENGTH; start_rank++) {
        //     int ses_itr{0};
        for (int res_itr = 0; res_itr < MAX_SERP_LENGTH - this->rank - 1; res_itr++) {
            log_prob += __logf(this->click_probs[this->rank + 1][res_itr]);
        }
        // }

        // for (int sub_rank = 0; sub_rank < MAX_SERP_LENGTH; sub_rank++) { // ! this part is most likely incorrect since print 5 always the wrong return value. replace this parclick with this snippet and change until it is correct.
        //     log_prob += __logf(this->click_probs[this->rank + 1][sub_rank]);
        // }
    }

    float exam_val = this->exam_probs[this->rank];

    if (x == 1) {
        log_prob += __logf(exam_val);
    }
    else {
        log_prob += __logf(1 - exam_val);
    }

    // printf("%d, %d] (%d,%d,%d) 5, log_prob = %f, returned = %f, [attr = %f, tau 1 = %f, tau 2 = %f, tau 3 = %f, last_click_rank = %d, rank = %d, click = %d\n", qid, did, x, y, z, log_prob, __expf(log_prob), this->attr, this->tau_1, this->tau_2, this->tau_3, this->last_click_rank, this->rank, this->click);
    return __expf(log_prob);
}

DEV DBNFactor::DBNFactor(float (&click_probs)[MAX_SERP_LENGTH][MAX_SERP_LENGTH], float (&exam_probs)[MAX_SERP_LENGTH + 1], int click,
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

DEV float DBNFactor::compute(int x, int y, int z) {
    float log_prob = 0.f;

    if (this->click == 0){
        if (y == 1){
            return 0.f;
        }

        log_prob += __logf(1 - this->attr);

        if (x == 1) {
            if (z == 1) {
                log_prob += __logf(this->gamma);
            }
            else {
                log_prob += __logf(1 - this->gamma);
            }
        }
        else if (z == 1) {
            return 0.f;
        }
    }
    else {
        if (x == 0) {
            return 0.f;
        }

        log_prob += __logf(this->attr);

        if (y == 0){
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
    }

    if (z == 0) {
        if (this->last_click_rank >= this->rank + 1) {
            return 0.f;
        }
    }
    else if (this->rank + 1 < MAX_SERP_LENGTH) {
        for (int sub_rank = 0; sub_rank < MAX_SERP_LENGTH; sub_rank++) {
            log_prob += __logf(this->click_probs[this->rank + 1][sub_rank]);
        }
    }

    float exam_val = this->exam_probs[this->rank];

    if (x == 1) {
        log_prob += __logf(exam_val);
    }
    else {
        log_prob += __logf(1 - exam_val);
    }

    return __expf(log_prob);
}