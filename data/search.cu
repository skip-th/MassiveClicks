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

HST SearchResult_Hst::SearchResult_Hst() = default;

/**
 * @brief Construct a new Search Result object.
 *
 * @param doc_id The document ID of the search result.
 * @param click Whether the document has been clicked (1) or not (0).
 * @param param The unique parameter index associated with all search results
 * with this query-document pair.
 */
HST SearchResult_Hst::SearchResult_Hst(const int& doc_id, const int& click, const int& param) {
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
HST void SearchResult_Hst::update_click(const int& click_val) {
    this->click = click_val;
}

/**
 * @brief Gets the document ID associated with this search result.
 *
 * @return The document ID of the search result.
 */
HST int SearchResult_Hst::get_doc_id() const{
    return doc_id;
}

/**
 * @brief Gets whether the search result has been clicked or not.
 *
 * @return Value representing a click (1) or no click (0) on the search
 * result.
 */
HST int SearchResult_Hst::get_click() const{
    return click;
}

/**
 * @brief Sets the value of the parameter index attribute of the search result.
 * The parameter index is a unique number associated with all search results
 * with the same query-document pair.
 *
 * @param index The index of this search result's parameter unqiue to all
 * search results with the same query-document pair.
 */
HST void SearchResult_Hst::set_param_index(const int& index) {
    this->param_index = index;
}

/**
 * @brief Returns the parameter index attribute of the search result.
 * The parameter index is a unique number associated with all search results
 * with the same query-document pair.
 *
 * @return The parameter index.
 */
HST int SearchResult_Hst::get_param_index() const{
    return param_index;
}


//---------------------------------------------------------------------------//
// A search engine result page (query) containing 10 documents.              //
//---------------------------------------------------------------------------//
HST SERP_Hst::SERP_Hst() = default;

/**
 * @brief Construct a new SERP_Hst object.
 *
 * @param line A raw dataset line representing a query of length 15.
 */
HST SERP_Hst::SERP_Hst(const std::vector<std::string>& line) {
    query = std::stoi(line[3]);

    for(int i = 0; i < MAX_SERP; i++) {
        session[i] = SearchResult_Hst(std::stoi(line[i + 5]), 0, -1);
    }
}

/**
 * @brief Update the click attribute of a search result with the same document
 * ID.
 *
 * @param line A raw dataset line representing a click on a document of length
 * 4.
 * @return true, if the document click attribute has succesfully been updated,
 * or false, if no document with the given document ID was found, and no search
 * result was updated.
 */
HST bool SERP_Hst::update_click_res(const std::vector<std::string>& line){
    int doc_id = std::stoi(line[3]);

    for (int j = 0; j < MAX_SERP; j++) {
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
 * @return The copy of the search result.
 */
HST SearchResult_Hst SERP_Hst::operator[] (const int& rank) const{
    return session[rank];
}

/**
 * @brief The query ID of the query session.
 *
 * @return The query ID.
 */
HST int SERP_Hst::get_query() const{
    return this->query;
}

/**
 * @brief Get a reference to a search result at a given rank within this
 * session. This is not a copy!
 *
 * @param rank The rank of the search result.
 * @return The reference to the search result within this session.
 */
HST SearchResult_Hst& SERP_Hst::access_sr(const int& rank) {
    return session[rank];
}

/**
 * @brief Retrieve the last rank at which a document has been clicked in this
 * SERP_Hst. The last rank (9) is returned if no clicked documents are found.
 *
 * @return The rank of the last clicked document.
 */
HST int SERP_Hst::last_click_rank(void) {
    int last_click_rank = MAX_SERP;

    for (int rank = MAX_SERP - 1; rank >= 0; --rank) {
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
 * occured for each document in this SERP_Hst. Documents without a previous click
 * get assigned the last rank (9).
 *
 * @param prev_click_rank The array which stores the rank of the previous click.
 * @return The clicks on each of the documents in this
 * query session.
 */
HST void SERP_Hst::prev_clicked_rank(int (&prev_click_rank)[MAX_SERP]) {
    int last_click_rank{MAX_SERP - 1};

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
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

DEV HST SearchResult_Dev::SearchResult_Dev() = default;

/**
 * @brief Construct a new Search Result object.
 *
 * @param click Whether the document has been clicked (1) or not (0).
 * @param param The unique parameter index associated with all search results
 * with this query-document pair.
 */
HST SearchResult_Dev::SearchResult_Dev(const int& click, const int& param) {
    this->click = static_cast<char>(click);
    this->param_index = param;
}

/**
 * @brief Gets whether the search result has been clicked or not.
 *
 * @return Integer representing a click (1) or no click (0) on the search
 * result.
 */
DEV int SearchResult_Dev::get_click() const{
    return click;
}

/**
 * @brief Returns the parameter index attribute of the search result.
 * The parameter index is a unique number associated with all search results
 * with the same query-document pair.
 *
 * @return The parameter index.
 */
DEV int SearchResult_Dev::get_param_index() const{
    return param_index;
}


//---------------------------------------------------------------------------//
// A search engine result page (query) containing 10 documents.              //
//---------------------------------------------------------------------------//

DEV HST SERP_Dev::SERP_Dev() = default;

/**
 * @brief Get a copy of a search result at a given rank within this session.
 *
 * @param rank The rank of the search result.
 * @return The copy of the search result.
 */
HST SearchResult_Dev SERP_Dev::operator[] (const int& rank) const{
    return session[rank];
}

/**
 * @brief Construct a new SERP_Dev object from a SERP_Hst object.
 *
 * @param serp A SERP_Hst object.
 */
HST SERP_Dev::SERP_Dev(const SERP_Hst serp) {
    for (int i = 0; i < MAX_SERP; i++) {
        SearchResult_Hst serp_tmp = serp[i];
        session[i] = SearchResult_Dev(serp_tmp.get_click(), serp_tmp.get_param_index());
    }
}

/**
 * @brief Construct a new SERP_Dev object from a SearchResult_Dev object array.
 *
 * @param dataset A SearchResult_Dev object array.
 * @param dataset_size The size of the SearchResult_Dev object array.
 * @param thread_index The index of the thread.
 */
DEV SERP_Dev::SERP_Dev(SearchResult_Dev*& dataset, int& dataset_size, int& thread_index) {
    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
        this->session[rank] = dataset[rank * dataset_size + thread_index];
    }
}

/**
 * @brief Retrieve the last rank at which a document has been clicked in this
 * SERP_Hst. The last rank (9) is returned if no clicked documents are found.
 *
 * @return The rank of the last clicked document.
 */
DEV int SERP_Dev::last_click_rank(void) {
    int last_click_rank = MAX_SERP;

    for (int rank = MAX_SERP - 1; rank >= 0; --rank) {
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
 * occured for each document in this SERP_Hst. Documents without a previous click
 * get assigned the last rank (9).
 *
 * @param prev_click_rank The array which stores the rank of the previous click.
 * @return The clicks on each of the documents in this query session.
 */
DEV void SERP_Dev::prev_clicked_rank(int (&prev_click_rank)[MAX_SERP]) {
    int last_click_rank{MAX_SERP - 1};

    #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
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
// Helper function for converting a SERP_Hst array to a SERP_Dev array.      //
//---------------------------------------------------------------------------//

/**
 * @brief Convert a host-side array of SERP_Hst objects to a smaller
 * SearchResult_Dev array which can be used on the device. Also change the
 * indexing scheme so that "neighboring" threads read from contiguous memory.
 *
 * @param dataset_hst The host-side array of SERP_Hst objects.
 * @param dataset_dev The SERP_Dev array which will be transfered to the
 * device.
 */
HST void convert_to_device(std::vector<SERP_Hst>& dataset_hst, std::vector<SearchResult_Dev>& dataset_dev) {
    dataset_dev.resize(dataset_hst.size() * MAX_SERP);
    int n_queries = dataset_hst.size();

    // Convert the host-side dataset to a smaller device-side dataset.
    for (int query_index = 0; query_index < dataset_hst.size(); query_index++) {
        SERP_Dev serp_tmp = SERP_Dev(dataset_hst[query_index]);
        #pragma unroll
    for (int rank = 0; rank < MAX_SERP; rank++) {
            // Change the indexing scheme so separate threads read from
            // contiguous memory.
            dataset_dev[rank * n_queries + query_index] = serp_tmp[rank];
        }
    }
}