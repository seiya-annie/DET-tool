package main

// Built-in query sets for external benchmarks (TPCC / TPCH)
// These queries are read-only and safe to run with EXPLAIN ANALYZE.

// GetTPCHQueries returns SQL statements for given TPCH query ids (e.g., q1, q6, q12).
// If ids is empty, defaults to a small representative set.
func GetTPCHQueries(ids []string) []string {
    // Minimal TPC-H reference queries adapted for TiDB/MySQL dialect
    tpch := map[string]string{
        "q1": `SELECT
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM
    lineitem
WHERE
    l_shipdate <= DATE_SUB('1998-12-01', INTERVAL 90 DAY)
GROUP BY
    l_returnflag, l_linestatus
ORDER BY
    l_returnflag, l_linestatus`,
        "q6": `SELECT
    SUM(l_extendedprice * l_discount) AS revenue
FROM
    lineitem
WHERE
    l_shipdate >= '1994-01-01'
    AND l_shipdate < DATE_ADD('1994-01-01', INTERVAL 1 YEAR)
    AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01
    AND l_quantity < 24`,
        "q12": `SELECT
    l_shipmode,
    SUM(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
    SUM(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
FROM
    orders o
JOIN
    lineitem l ON o.o_orderkey = l.l_orderkey
WHERE
    l_shipmode IN ('MAIL', 'SHIP')
    AND l_commitdate < l_receiptdate
    AND l_shipdate < l_commitdate
    AND l_receiptdate >= '1994-01-01'
    AND l_receiptdate < DATE_ADD('1994-01-01', INTERVAL 1 YEAR)
GROUP BY
    l_shipmode
ORDER BY
    l_shipmode`,
    }

    if len(ids) == 0 {
        ids = []string{"q1", "q6", "q12"}
    }

    var out []string
    for _, id := range ids {
        if sql, ok := tpch[id]; ok {
            out = append(out, sql)
        }
    }
    return out
}

// GetTPCCQueries returns a small set of read-only queries over TPCC schema.
// These are not official TPC-C transactions, but analytical/read queries to evaluate optimizer behavior.
func GetTPCCQueries() []string {
    return []string{
        `SELECT COUNT(*) FROM customer`,
        `SELECT c_w_id, c_d_id, COUNT(*) AS cnt FROM customer GROUP BY c_w_id, c_d_id`,
        `SELECT /*+ HASH_JOIN(o c) */ o_w_id, o_d_id, COUNT(*) AS orders_cnt FROM orders o JOIN customer c ON o.o_w_id=c.c_w_id AND o.o_d_id=c.c_d_id AND o.o_c_id=c.c_id GROUP BY o_w_id, o_d_id`,
        `SELECT SUM(ol_amount) FROM order_line WHERE ol_delivery_d >= DATE_SUB(NOW(), INTERVAL 30 DAY)`,
        `SELECT i_id, COUNT(*) AS cnt FROM stock s JOIN item i ON s.s_i_id = i.i_id GROUP BY i_id ORDER BY cnt DESC LIMIT 50`,
    }
}

