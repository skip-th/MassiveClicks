/** Click model factor.
 *
 * factor.cuh:
 *  - Defines the functions for processing the phi formula of CCM and DBN.
 */

#ifndef CLICK_MODEL_FACTOR_H
#define CLICK_MODEL_FACTOR_H

#include "../utils/definitions.h"
#include "base.cuh"

class CCMFactor {
private:
    int click, last_click_rank, rank;
    float attr, tau_1, tau_2, tau_3;
    float (*click_probs)[MAX_SERP];
    float (*exam_probs);
public:
    HST DEV CCMFactor(float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1], int click, int last_click_rank, int rank, float attr, float tau_1, float tau_2, float tau_3);
    HST DEV float compute(int x, int y, int z);
};


class DBNFactor {
private:
    int click, last_click_rank, rank;
    float attr, sat, gamma;
    float (*click_probs)[MAX_SERP];
    float (*exam_probs);
public:
    HST DEV DBNFactor(float (&click_probs)[MAX_SERP][MAX_SERP], float (&exam_probs)[MAX_SERP + 1], int click, int last_click_rank, int rank, float attr, float sat, float gamma);
    HST DEV float compute(int x, int y, int z);
};


#endif // CLICK_MODEL_FACTOR_H
