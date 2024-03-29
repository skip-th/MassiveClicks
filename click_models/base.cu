/** First implementation of a generalized CM base.
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
 * @param model_type The type of click model (e.g., 0 = PBM).
 * @return The click model object of a given model type.
 */
HST ClickModel_Hst* create_cm_host(const int model_type) {
    ClickModel_Hst *cm_temp;
    switch (model_type) {
        case 0:{
            PBM_Hst pbm_model;
            cm_temp = &pbm_model;
            break;
        }
        case 1:{
            CCM_Hst ccm_model;
            cm_temp = &ccm_model;
            break;
        }
        case 2:{
            DBN_Hst dbn_model;
            cm_temp = &dbn_model;
            break;
        }
        case 3:{
            UBM_Hst ubm_model;
            cm_temp = &ubm_model;
            break;
        }
        default: {
            PBM_Hst pbm_is_default;
            cm_temp = &pbm_is_default;
            break;
        }
    }
    return cm_temp->clone();
}


/**
 * @brief Create a click model object for the GPU device.
 *
 * @param model_type The type of click model (e.g., 0 = PBM).
 * @return The click model object of a given model type.
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
