/** Click model session classes.
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

// A SearchResult_Hst contains a query-document pair from within a session/SERP_Hst.
class SearchResult_Hst {
public:
    HST SearchResult_Hst();
    HST SearchResult_Hst(const int& doc_id, const int& click, const int& param);

    HST void update_click(const int& click_val);
    HST void set_param_index(const int& index);
    HST int get_doc_id() const;
    HST int get_click() const;
    HST int get_param_index() const;
private:
    int click, doc_id, param_index;
};

// A Search Engine Result Page (SERP_Hst). A set of 10 documents (SearchResult_Hst)
// related to a given query, and whether those documents have been clicked.
class SERP_Hst {
public:
    HST SERP_Hst();
    HST explicit SERP_Hst(const std::vector<std::string>& line);
    HST SearchResult_Hst operator[] (const int& rank) const;

    HST bool update_click_res(const std::vector<std::string>& line);
    HST int get_query() const;
    HST void prev_clicked_rank(int (&prev_click_rank)[MAX_SERP]);
    HST int last_click_rank(void);
    HST SearchResult_Hst& access_sr(const int& rank);

private:
    int query{-1};
    SearchResult_Hst session[MAX_SERP]{};
};


//---------------------------------------------------------------------------//
// Device-side functions.                                                    //
//---------------------------------------------------------------------------//

// A SearchResult_Dev contains a query-document pair from within a session/SERP.
class SearchResult_Dev {
public:
    DEV HST SearchResult_Dev();
    HST SearchResult_Dev(const int& click, const int& param);

    DEV int get_click() const;
    DEV int get_param_index() const;
private:
    char click;
    int param_index;
};

// A Search Engine Result Page (SERP). A set of 10 documents (SearchResult_Dev)
// related to a given query, and whether those documents have been clicked.
class SERP_Dev {
public:
    DEV HST SERP_Dev();
    HST explicit SERP_Dev(const SERP_Hst serp);
    DEV explicit SERP_Dev(SearchResult_Dev*& dataset, int& dataset_size, int& thread_index);
    HST DEV SearchResult_Dev operator[] (const int& rank) const;
    DEV void prev_clicked_rank(int (&prev_click_rank)[MAX_SERP]);
    DEV int last_click_rank(void);
private:
    SearchResult_Dev session[MAX_SERP]{};
};

HST void convert_to_device(std::vector<SERP_Hst>& dataset_hst, std::vector<SearchResult_Dev>& dataset_dev);

#endif // CLICK_MODEL_SESSION_H