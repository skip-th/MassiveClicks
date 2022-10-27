/** Click model session classes.
 * Pooya Khandel's ParClick is used as a reference implementation.
 *
 * session.h:
 *  - Declare the Session and SubSession classes.
 */

// Use header guards to prevent the header from being included multiple times.
#ifndef CLICK_MODEL_SESSION_H
#define CLICK_MODEL_SESSION_H

// System include.
#include <string>
#include <vector>
#include <array>

// User include.
#include "../utils/definitions.h"

//---------------------------------------------------------------------------//
// Host-side functions.                                                      //
//---------------------------------------------------------------------------//

// A SearchResult contains a query-document pair from within a session/SERP.
class SearchResult {
public:
    HST SearchResult();
    HST SearchResult(const int& doc_id, const int& click, const int& param);

    HST void update_click(const int& click_val);
    HST void set_param_index(const int& index);
    HST int get_doc_id() const;
    HST int get_click() const;
    HST int get_param_index() const;
private:
    int click;
    int doc_id, param_index;
};

// A Search Engine Result Page (SERP). A set of 10 documents (SearchResult)
// related to a given query, and whether those documents have been clicked.
class SERP {
public:
    HST SERP();
    HST explicit SERP(const std::vector<std::string>& line);
    HST SearchResult operator[] (const int& rank) const;

    HST bool update_click_res(const std::vector<std::string>& line);
    HST int get_query() const;
    HST void prev_clicked_rank(int (&prev_click_rank)[MAX_SERP_LENGTH]);
    HST int last_click_rank(void);
    HST SearchResult& access_sr(const int& rank);

private:
    int query{-1};
    SearchResult session[MAX_SERP_LENGTH]{};
};

//---------------------------------------------------------------------------//
// Device-side functions.                                                    //
//---------------------------------------------------------------------------//

// A SearchResult contains a query-document pair from within a session/SERP.
class SearchResult_DEV {
public:
    HST SearchResult_DEV();
    HST SearchResult_DEV(const int& click, const int& param);

    DEV int get_click() const;
    DEV int get_param_index() const;
private:
    char click;
    int param_index;
};

// A Search Engine Result Page (SERP). A set of 10 documents (SearchResult)
// related to a given query, and whether those documents have been clicked.
class SERP_DEV {
public:
    HST SERP_DEV();
    HST explicit SERP_DEV(const SERP serp);
    HST DEV SearchResult_DEV operator[] (const int& rank) const;
    DEV void prev_clicked_rank(int (&prev_click_rank)[MAX_SERP_LENGTH]);
    DEV int last_click_rank(void);
private:
    SearchResult_DEV session[MAX_SERP_LENGTH]{};
};

HST void convert_to_device(std::vector<SERP>& dataset_hst, std::vector<SERP_DEV>& dataset_dev);

#endif // CLICK_MODEL_SESSION_H