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

// A SearchResult contains a query-document pair from within a session.
class SearchResult {
public:
    HST DEV SearchResult();
    SearchResult(const int& doc_id, const int& doc_rank, const int& click, const int& param);

    HST void update_click(const int& click_val);
    HST void set_param_index(const int& index);
    HST DEV int get_doc_id() const;
    HST DEV int get_doc_rank() const;
    HST DEV int get_click() const;
    HST DEV int get_param_index() const;
private:
    int doc_id, doc_rank, click, param_index;
};

// A Search Engine Result Page (SERP). A set of 10 documents (SearchResult)
// related to a given query, and whether those documents have been clicked.
class SERP {
public:
    HST DEV SERP();
    HST explicit SERP(const std::vector<std::string>& line);

    HST bool update_click_res(const std::vector<std::string>& line);
    HST DEV SearchResult operator[] (const int& rank) const;
    HST int sid() const;
    HST DEV int get_query() const;
    HST DEV void prev_clicked_rank(int (&prev_click_rank)[MAX_SERP_LENGTH]);
    HST DEV int last_click_rank(void);
    HST SearchResult& access_sr(const int& rank);

private:
    int session_id{-1}, rank_doc{}, query{-1};
    SearchResult session[MAX_SERP_LENGTH]{};
};

#endif // CLICK_MODEL_SESSION_H