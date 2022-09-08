/** First implementation of a generalized CM base.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * base.cu:
 *  - Defines the generalized click model functions.
 */

#include "base.cuh"
#include "pbm.cuh"
#include "ccm.cuh"
#include "dbn.cuh"
#include "ubm.cuh"

/**
 * @brief Create a click model object for the host.
 *
 * @param model_type The type of click model (e.g. 0 = PBM).
 * @return ClickModel_Host* The click model object of a given model type.
 */
HST ClickModel_Host* create_cm_host(const int model_type) {
    ClickModel_Host *cm_temp;
    switch (model_type) {
        case 0:{
            PBM_Host pbm_model;
            cm_temp = &pbm_model;
            break;
        }
        case 1:{
            CCM_Host ccm_model;
            cm_temp = &ccm_model;
            break;
        }
        case 2:{
            DBN_Host dbn_model;
            cm_temp = &dbn_model;
            break;
        }
        case 3:{
            UBM_Host ubm_model;
            cm_temp = &ubm_model;
            break;
        }
        default: {
            PBM_Host pbm_is_default;
            cm_temp = &pbm_is_default;
            break;
        }
    }
    return cm_temp->clone();
}

/**
 * @brief Create a click model object for the GPU device.
 *
 * @param model_type The type of click model (e.g. 0 = PBM).
 * @return ClickModel_Dev* The click model object of a given model type.
 */
DEV ClickModel_Dev* create_cm_dev(const int model_type) {
    ClickModel_Dev *cm_temp;
    switch (model_type) {
        case 0:{
            PBM_Dev pbm_model;
            cm_temp = &pbm_model;
            break;
        }
        case 1:{
            CCM_Dev ccm_model;
            cm_temp = &ccm_model;
            break;
        }
        case 2:{
            DBN_Dev dbn_model;
            cm_temp = &dbn_model;
            break;
        }
        case 3:{
            UBM_Dev ubm_model;
            cm_temp = &ubm_model;
            break;
        }
        default: {
            PBM_Dev pbm_is_default;
            cm_temp = &pbm_is_default;
            break;
        }
    }
    return cm_temp->clone();
}
