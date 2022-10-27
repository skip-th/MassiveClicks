/** Click model session classes.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * session.cpp:
 *  - Defines the behaviour of the Session and SubSession classes.
 */

#include "search.cuh"

//---------------------------------------------------------------------------//
// Host-side functions.                                                      //
// A single search result (document) from a search result page (query).      //
//---------------------------------------------------------------------------//

HST SearchResult::SearchResult() = default;

/**
 * @brief Construct a new Search Result object.
 *
 * @param doc_id The document ID of the search result.
 * @param click Whether the document has been clicked (1) or not (0).
 * @param param The unique parameter index associated with all search results
 * with this query-document pair.
 */
HST SearchResult::SearchResult(const int& doc_id, const int& click, const int& param) {
    this->doc_id = doc_id;
    this->click = click;
    this->param_index = param;
}

/**
 * @brief Update the click attribute of a search result to 0 or 1.
 *
 * @param click_val New value of the click parameter. Can be either 0 (not
 * clicked) or 1 (clicked).
 */
HST void SearchResult::update_click(const int& click_val) {
    this->click = click_val;
}

/**
 * @brief Gets the document ID associated with this search result.
 *
 * @return int The document ID of the search result.
 */
HST int SearchResult::get_doc_id() const{
    return doc_id;
}

/**
 * @brief Gets whether the search result has been clicked or not.
 *
 * @return int Integer representing a click (1) or no click (0) on the search
 * result.
 */
HST int SearchResult::get_click() const{
    return click;
}

/**
 * @brief Sets the value of the parameter index attribute of the search result.
 * The parameter index is a unique number associated with all search results
 * with the same query-document pair.
 */
HST void SearchResult::set_param_index(const int& index) {
    this->param_index = index;
}

/**
 * @brief Returns the parameter index attribute of the search result.
 * The parameter index is a unique number associated with all search results
 * with the same query-document pair.
 *
 * @return int The parameter index.
 */
HST int SearchResult::get_param_index() const{
    return param_index;
}


//---------------------------------------------------------------------------//
// A search engine result page (query) containing 10 documents.              //
//---------------------------------------------------------------------------//
HST SERP::SERP() = default;

/**
 * @brief Construct a new SERP object.
 *
 * @param line A raw dataset line representing a query of length 15.
 */
HST SERP::SERP(const std::vector<std::string>& line) {
    query = std::stoi(line[3]);

    for(int i = 0; i < MAX_SERP_LENGTH; i++) {
        session[i] = SearchResult(std::stoi(line[i + 5]), 0, -1);
    }
}

/**
 * @brief Update the click attribute of a search result with the same document
 * ID.
 *
 * @param line A raw dataset line representing a click on a document of length
 * 4.
 * @return true The document click attribute has succesfully been updated.
 * @return false No document with the given document ID was found, so no search
 * result was updated.
 */
HST bool SERP::update_click_res(const std::vector<std::string>& line){
    int doc_id = std::stoi(line[3]);

    for (int j = 0; j < MAX_SERP_LENGTH; j++) {
        if (session[j].get_doc_id() == doc_id) {
            session[j].update_click(1);
            return true;
        }
    }

    return false;
}

/**
 * @brief Get a copy of a search result at a given rank within this session.
 *
 * @param rank The rank of the search result.
 * @return SearchResult The copy of the search result.
 */
HST SearchResult SERP::operator[] (const int& rank) const{
    return session[rank];
}

/**
 * @brief The query ID of the query session.
 *
 * @return int The query ID.
 */
HST int SERP::get_query() const{
    return this->query;
}

/**
 * @brief Get a reference to a search result at a given rank within this
 * session. This is not a copy!
 *
 * @param rank The rank of the search result.
 * @return SearchResult The reference to the search result within this session.
 */
HST SearchResult& SERP::access_sr(const int& rank) {
    return session[rank];
}

/**
 * @brief Retrieve the last rank at which a document has been clicked in this
 * SERP. The last rank (9) is returned if no clicked documents are found.
 *
 * @return int The rank of the last clicked document.
 */
HST int SERP::last_click_rank(void) {
    int last_click_rank = MAX_SERP_LENGTH;

    for (int rank = MAX_SERP_LENGTH - 1; rank >= 0; --rank) {
        // If the current rank has been clicked, the preceding ranks don't need
        // to be checked.
        if (this->session[rank].get_click() == 1) {
            last_click_rank = rank;
            break;
        }
    }

    return last_click_rank;
}

/**
 * @brief Retrieve an array which stores the previous rank at which a click has
 * occured for each document in this SERP. Documents without a previous click
 * get assigned the last rank (9).
 *
 * @return std::array<int, 10> The clicks on each of the documents in this
 * query session.
 */
HST void SERP::prev_clicked_rank(int (&prev_click_rank)[MAX_SERP_LENGTH]) {
    int last_click_rank{MAX_SERP_LENGTH - 1};

    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        // Store a previous click. The last rank will be stored if no click has
        // happened before.
        prev_click_rank[rank] = last_click_rank;

        // If the current rank has been clicked. Save it for the next ranks.
        if (this->session[rank].get_click() == 1) {
            last_click_rank = rank;
        }
    }
}

//---------------------------------------------------------------------------//
// Device-side functions.                                                    //
// A single search result (document) from a search result page (query).      //
//---------------------------------------------------------------------------//

HST SearchResult_DEV::SearchResult_DEV() = default;

/**
 * @brief Construct a new Search Result object.
 *
 * @param click Whether the document has been clicked (1) or not (0).
 * @param param The unique parameter index associated with all search results
 * with this query-document pair.
 */
HST SearchResult_DEV::SearchResult_DEV(const int& click, const int& param) {
    this->click = static_cast<char>(click);
    this->param_index = param;
}

/**
 * @brief Gets whether the search result has been clicked or not.
 *
 * @return int Integer representing a click (1) or no click (0) on the search
 * result.
 */
DEV int SearchResult_DEV::get_click() const{
    return click;
}

/**
 * @brief Returns the parameter index attribute of the search result.
 * The parameter index is a unique number associated with all search results
 * with the same query-document pair.
 *
 * @return int The parameter index.
 */
DEV int SearchResult_DEV::get_param_index() const{
    return param_index;
}


//---------------------------------------------------------------------------//
// A search engine result page (query) containing 10 documents.              //
//---------------------------------------------------------------------------//

HST SERP_DEV::SERP_DEV() = default;

/**
 * @brief Get a copy of a search result at a given rank within this session.
 *
 * @param rank The rank of the search result.
 * @return SearchResult_DEV The copy of the search result.
 */
HST SearchResult_DEV SERP_DEV::operator[] (const int& rank) const{
    return session[rank];
}

/**
 * @brief Construct a new SERP_DEV object from a SERP_HST object.
 *
 * @param serp A SERP_HST object.
 */
HST SERP_DEV::SERP_DEV(const SERP serp) {
    for (int i = 0; i < MAX_SERP_LENGTH; i++) {
        SearchResult serp_tmp = serp[i];
        session[i] = SearchResult_DEV(serp_tmp.get_click(), serp_tmp.get_param_index());
    }
}

/**
 * @brief Retrieve the last rank at which a document has been clicked in this
 * SERP. The last rank (9) is returned if no clicked documents are found.
 *
 * @return int The rank of the last clicked document.
 */
DEV int SERP_DEV::last_click_rank(void) {
    int last_click_rank = MAX_SERP_LENGTH;

    for (int rank = MAX_SERP_LENGTH - 1; rank >= 0; --rank) {
        // If the current rank has been clicked, the preceding ranks don't need
        // to be checked.
        if (this->session[rank].get_click() == 1) {
            last_click_rank = rank;
            break;
        }
    }

    return last_click_rank;
}

/**
 * @brief Retrieve an array which stores the previous rank at which a click has
 * occured for each document in this SERP. Documents without a previous click
 * get assigned the last rank (9).
 *
 * @return std::array<int, 10> The clicks on each of the documents in this
 * query session.
 */
DEV void SERP_DEV::prev_clicked_rank(int (&prev_click_rank)[MAX_SERP_LENGTH]) {
    int last_click_rank{MAX_SERP_LENGTH - 1};

    for (int rank = 0; rank < MAX_SERP_LENGTH; rank++) {
        // Store a previous click. The last rank will be stored if no click has
        // happened before.
        prev_click_rank[rank] = last_click_rank;

        // If the current rank has been clicked. Save it for the next ranks.
        if (this->session[rank].get_click() == 1) {
            last_click_rank = rank;
        }
    }
}


//---------------------------------------------------------------------------//
// Helper function for converting a SERP_HST array to a SERP_DEV array.      //
//---------------------------------------------------------------------------//

/**
 * @brief Convert a host-side array of SERP objects to a smaller SERP_DEV array
 * whcih can be used on the device.
 *
 * @param dataset_hst The host-side array of SERP objects.
 * @param dataset_dev The SERP_DEV array which will be transfered to the
 * device.
 */
HST void convert_to_device(std::vector<SERP>& dataset_hst, std::vector<SERP_DEV>& dataset_dev) {
    // Convert the host-side dataset to a smaller device-side dataset.
    dataset_dev.resize(dataset_hst.size());
    for (int i = 0; i < dataset_hst.size(); i++) {
        dataset_dev[i] = SERP_DEV(dataset_hst[i]);
    }
}