from collections import defaultdict
from pathlib import Path

import numpy as np

from .csv_utils import (
    BLE_BLE_LINK_RESULT_FIELDS,
    BLE_BLE_PAIR_SCORE_FIELDS,
    CLASSIFICATION_REPORT_FIELDS,
    LINK_RESULT_FIELDS,
    PAIR_SCORE_FIELDS,
    read_csv,
    write_csv,
)
from .plotting import plot_best_pairs_hist, plot_offset_scatter, plot_rmse_ecdf
from .plotting import plot_all_pair_offsets
from .plotting import (
    plot_ble_ble_all_pair_offsets,
    plot_ble_ble_best_pairs_hist,
    plot_ble_ble_offset_scatter,
    plot_ble_ble_rmse_ecdf,
)


def wrap_to_bluetooth_slot(delta_us, slot_us=625.0):
    if slot_us <= 0:
        raise ValueError("slot_us must be greater than zero")
    half = slot_us / 2.0
    return ((np.asarray(delta_us, dtype=float) + half) % slot_us) - half


def group_rows(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(key, "")
        if value:
            grouped[value].append(row)
    return grouped


def normalize_hex_id(value, digits):
    text = str(value or "").strip().replace("0x", "").replace("0X", "").upper()
    text = "".join(ch for ch in text if ch in "0123456789ABCDEF")
    if not text:
        return ""
    return text[-digits:].zfill(digits)


def hamming_distance_bits(left, right):
    if not left or not right or len(left) != len(right):
        return 999
    return (int(left, 16) ^ int(right, 16)).bit_count()


def tolerant_group_rows(rows, key, digits, max_bit_errors):
    counts = defaultdict(int)
    normalized_rows = []
    for row in rows:
        normalized = normalize_hex_id(row.get(key, ""), digits)
        if not normalized:
            continue
        normalized_rows.append((normalized, row))
        counts[normalized] += 1

    representatives = []
    for value, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if any(hamming_distance_bits(value, rep) <= max_bit_errors for rep in representatives):
            continue
        representatives.append(value)

    grouped = defaultdict(list)
    aliases = defaultdict(lambda: defaultdict(int))
    for value, row in normalized_rows:
        if max_bit_errors <= 0:
            representative = value
        else:
            representative = min(
                representatives,
                key=lambda rep: (hamming_distance_bits(value, rep), -counts[rep], rep),
            )
            if hamming_distance_bits(value, representative) > max_bit_errors:
                representative = value
        label = f"0x{representative}" if digits == 8 else representative
        grouped[label].append(row)
        aliases[label][value] += 1
    return grouped, aliases


def format_aliases(alias_counts, digits):
    parts = []
    for value, count in sorted(alias_counts.items(), key=lambda item: (-item[1], item[0])):
        label = f"0x{value}" if digits == 8 else value
        parts.append(f"{label}:{count}")
    return ";".join(parts)


def exact_group_rows(rows, key, digits):
    grouped = defaultdict(list)
    aliases = defaultdict(lambda: defaultdict(int))
    for row in rows:
        normalized = normalize_hex_id(row.get(key, ""), digits)
        if not normalized:
            continue
        label = normalized
        grouped[label].append(row)
        aliases[label][normalized] += 1
    return grouped, aliases


def filter_groups_by_min_occurrences(groups, aliases=None, min_occurrences=1):
    if min_occurrences <= 1:
        return groups, aliases
    filtered = {
        key: rows for key, rows in groups.items() if len(rows) >= min_occurrences
    }
    if aliases is None:
        return filtered, None
    filtered_aliases = {
        key: alias_counts for key, alias_counts in aliases.items() if key in filtered
    }
    return filtered, filtered_aliases


def ble_display_id(row):
    return row.get("ble_device_address") or row.get("access_address", "")


def build_ble_groups(ble_rows, ble_aa_tolerance_bits, min_id_occurrences=1):
    adv_rows = [row for row in ble_rows if row.get("ble_device_address")]
    aa_rows = [row for row in ble_rows if not row.get("ble_device_address")]

    adv_groups, adv_aliases = exact_group_rows(adv_rows, "ble_device_address", 12)
    aa_groups, aa_aliases = tolerant_group_rows(aa_rows, "access_address", 8, ble_aa_tolerance_bits)
    adv_groups, adv_aliases = filter_groups_by_min_occurrences(
        adv_groups, adv_aliases, min_id_occurrences
    )
    aa_groups, aa_aliases = filter_groups_by_min_occurrences(
        aa_groups, aa_aliases, min_id_occurrences
    )

    groups = []
    for device_address, rows in sorted(adv_groups.items()):
        access_counts = defaultdict(int)
        for row in rows:
            access_counts[row.get("access_address", "")] += 1
        ble_access_address = ""
        if access_counts:
            ble_access_address = max(access_counts.items(), key=lambda item: (item[1], item[0]))[0]
        groups.append(
            {
                "group_id": f"dev:{device_address}",
                "rows": rows,
                "ble_access_address": ble_access_address,
                "ble_access_address_aliases": format_aliases(
                    {normalize_hex_id(value, 8): count for value, count in access_counts.items() if value},
                    8,
                ),
                "ble_device_address": device_address,
                "ble_device_address_aliases": format_aliases(adv_aliases[device_address], 12),
                "display_id": device_address,
            }
        )

    for ble_aa, rows in sorted(aa_groups.items()):
        groups.append(
            {
                "group_id": f"aa:{ble_aa}",
                "rows": rows,
                "ble_access_address": ble_aa,
                "ble_access_address_aliases": format_aliases(aa_aliases[ble_aa], 8),
                "ble_device_address": "",
                "ble_device_address_aliases": "",
                "display_id": ble_aa,
            }
        )
    return groups


def build_ble_adv_conn_groups(ble_rows, ble_aa_tolerance_bits, min_id_occurrences=1):
    adv_rows = [row for row in ble_rows if row.get("ble_device_address")]
    conn_rows = [row for row in ble_rows if not row.get("ble_device_address") and row.get("access_address")]

    adv_groups, adv_aliases = exact_group_rows(adv_rows, "ble_device_address", 12)
    conn_groups, conn_aliases = tolerant_group_rows(conn_rows, "access_address", 8, ble_aa_tolerance_bits)
    adv_groups, adv_aliases = filter_groups_by_min_occurrences(
        adv_groups, adv_aliases, min_id_occurrences
    )
    conn_groups, conn_aliases = filter_groups_by_min_occurrences(
        conn_groups, conn_aliases, min_id_occurrences
    )

    adv_group_list = []
    for advertiser_address, rows in sorted(adv_groups.items()):
        adv_group_list.append(
            {
                "group_id": f"adv:{advertiser_address}",
                "advertiser_address": advertiser_address,
                "advertiser_address_aliases": format_aliases(adv_aliases[advertiser_address], 12),
                "rows": rows,
            }
        )

    conn_group_list = []
    for access_address, rows in sorted(conn_groups.items()):
        conn_group_list.append(
            {
                "group_id": f"conn:{access_address}",
                "connection_access_address": access_address,
                "connection_access_address_aliases": format_aliases(conn_aliases[access_address], 8),
                "rows": rows,
            }
        )
    return adv_group_list, conn_group_list


def row_times_us(rows):
    return np.asarray([float(row["timestamp_us"]) for row in rows if row.get("timestamp_us") not in ("", None)])


def find_nearest_packets(source_times_us, target_times_us, search_window_us):
    target = np.sort(np.asarray(target_times_us, dtype=float))
    pairs = []
    if target.size == 0:
        return pairs
    for t_src in np.asarray(source_times_us, dtype=float):
        pos = np.searchsorted(target, t_src)
        candidates = []
        if pos < target.size:
            candidates.append(target[pos])
        if pos > 0:
            candidates.append(target[pos - 1])
        if not candidates:
            continue
        nearest = min(candidates, key=lambda t: abs(t_src - t))
        if abs(t_src - nearest) <= search_window_us:
            pairs.append((t_src, nearest))
    return pairs


def compute_pair_offsets(ble_times_us, bt_times_us, slot_us, search_window_us):
    pairs = find_nearest_packets(ble_times_us, bt_times_us, search_window_us)
    if not pairs:
        return np.empty(0), np.empty(0), []
    deltas = np.asarray([ble_t - bt_t for ble_t, bt_t in pairs], dtype=float)
    wrapped = wrap_to_bluetooth_slot(deltas, slot_us)
    ble_times = np.asarray([ble_t for ble_t, _ in pairs], dtype=float)
    return wrapped, ble_times, pairs


def estimate_mode_offset(wrapped_delta_us, bin_width_us=2.0):
    values = np.asarray(wrapped_delta_us, dtype=float)
    if values.size == 0:
        return np.nan
    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        return low
    bins = np.arange(low, high + bin_width_us, bin_width_us)
    if bins.size < 2:
        return float(np.median(values))
    counts, edges = np.histogram(values, bins=bins)
    idx = int(np.argmax(counts))
    in_bin = values[(values >= edges[idx]) & (values < edges[idx + 1])]
    if in_bin.size == 0:
        return float((edges[idx] + edges[idx + 1]) / 2.0)
    return float(np.median(in_bin))


def compute_link_score(wrapped_delta_us, slot_us=625.0, outlier_window_us=0.0):
    values = np.asarray(wrapped_delta_us, dtype=float)
    if values.size == 0:
        return {
            "valid_pair_count": 0,
            "rmse_us": np.nan,
            "rmse_demean_us": np.nan,
            "mode_delta_us": np.nan,
            "mean_delta_us": np.nan,
            "median_delta_us": np.nan,
            "std_delta_us": np.nan,
            "mad_us": np.nan,
            "iqr_us": np.nan,
            "filtered_pair_count": 0,
        }
    mode = estimate_mode_offset(values)
    median = float(np.median(values))

    # Outlier rejection: keep only pairs within outlier_window_us of mode
    if outlier_window_us > 0:
        dist_from_mode = np.abs(wrap_to_bluetooth_slot(values - mode, slot_us))
        mask = dist_from_mode <= outlier_window_us
        filtered = values[mask]
        if filtered.size >= 3:
            values = filtered
            mode = estimate_mode_offset(values)
            median = float(np.median(values))

    demeaned = wrap_to_bluetooth_slot(values - mode, slot_us)
    return {
        "valid_pair_count": int(values.size),
        "rmse_us": float(np.sqrt(np.mean(values**2))),
        "rmse_demean_us": float(np.sqrt(np.mean(demeaned**2))),
        "mode_delta_us": mode,
        "mean_delta_us": float(np.mean(values)),
        "median_delta_us": median,
        "std_delta_us": float(np.std(values)),
        "mad_us": float(np.median(np.abs(values - median))),
        "iqr_us": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "filtered_pair_count": int(values.size),
    }


def make_link_decision(score, score_threshold_us, min_valid_pairs, ambiguous=False):
    if score["valid_pair_count"] < min_valid_pairs:
        return "insufficient_packets"
    if score["rmse_demean_us"] <= score_threshold_us:
        return "candidate_ambiguous" if ambiguous else "candidate_link"
    return "no_link"


def circular_concentration(wrapped_delta_us, slot_us=625.0):
    """Measure how tightly deltas cluster around a single value.

    Returns a value in [0, 1]: 1 = perfectly concentrated, 0 = uniformly spread.
    Uses 1 - circular_variance, where deltas are mapped to angles on [0, 2pi).
    """
    values = np.asarray(wrapped_delta_us, dtype=float)
    if values.size < 2:
        return 1.0 if values.size == 1 else 0.0
    angles = 2.0 * np.pi * (values + slot_us / 2.0) / slot_us
    cos_mean = np.mean(np.cos(angles))
    sin_mean = np.mean(np.sin(angles))
    R = np.sqrt(cos_mean**2 + sin_mean**2)
    return float(R)


def classify_links(
    pair_scores,
    pair_offset_payloads,
    score_threshold_us=15.0,
    min_valid_pairs=5,
    slot_us=625.0,
    ble_group_key="_ble_group_id",
    rmse_key="rmse_demean_us",
):
    """Enrich pair_scores with auto-classification metrics.

    Adds: rank_in_ble_group, rmse_rel_score, gap_ratio_vs_runner_up,
    circular_concentration, auto_decision.
    Also generates a classification_report of promising candidates.
    """
    # Build lookup from pair_score row to wrapped deltas
    offset_lookup = {}
    for row, _ble_times, wrapped in pair_offset_payloads:
        offset_lookup[id(row)] = wrapped

    # Group by BLE group
    groups = defaultdict(list)
    for row in pair_scores:
        groups[row.get(ble_group_key, "")].append(row)

    report_rows = []

    for _group_id, rows in groups.items():
        # Parse rmse values, skip empty/inf
        scored = []
        for row in rows:
            val = row.get(rmse_key, "")
            if val == "" or val is None:
                continue
            try:
                fval = float(val)
            except (ValueError, TypeError):
                continue
            if np.isinf(fval) or np.isnan(fval):
                continue
            scored.append((row, fval))

        if not scored:
            for row in rows:
                row["rank_in_ble_group"] = ""
                row["rmse_rel_score"] = ""
                row["gap_ratio_vs_runner_up"] = ""
                row["circular_concentration"] = ""
                row["auto_decision"] = "insufficient_packets"
            continue

        # Sort by rmse ascending (best first)
        scored.sort(key=lambda x: x[1])
        rmse_values = np.array([s[1] for s in scored])
        rmse_median = float(np.median(rmse_values))
        rmse_std = float(np.std(rmse_values)) if rmse_values.size > 1 else 0.0

        for rank, (row, rmse_val) in enumerate(scored, start=1):
            row["rank_in_ble_group"] = rank
            # Relative score: how many std below median (negative = better)
            if rmse_std > 0:
                row["rmse_rel_score"] = fmt((rmse_val - rmse_median) / rmse_std)
            else:
                row["rmse_rel_score"] = fmt(0.0)

            # Gap ratio: only meaningful for rank 1
            if rank == 1 and len(scored) >= 2:
                runner_up_rmse = scored[1][1]
                if rmse_val > 0:
                    row["gap_ratio_vs_runner_up"] = fmt(runner_up_rmse / rmse_val)
                else:
                    row["gap_ratio_vs_runner_up"] = ""
            else:
                row["gap_ratio_vs_runner_up"] = ""

            # Circular concentration
            wrapped = offset_lookup.get(id(row), np.empty(0))
            conc = circular_concentration(wrapped, slot_us)
            row["circular_concentration"] = fmt(conc)

            # Auto decision
            vp = int(row.get("valid_pair_count", 0) or 0)
            if vp < min_valid_pairs:
                row["auto_decision"] = "insufficient_packets"
            elif rmse_val <= score_threshold_us:
                row["auto_decision"] = "candidate_link"
            elif rank == 1:
                rel = float(row["rmse_rel_score"]) if row["rmse_rel_score"] else 0.0
                gap = float(row["gap_ratio_vs_runner_up"]) if row["gap_ratio_vs_runner_up"] else 1.0
                if (rel < -1.0 or gap > 2.0) and conc > 0.5:
                    row["auto_decision"] = "possible_link"
                else:
                    row["auto_decision"] = "no_link"
            else:
                row["auto_decision"] = "no_link"

            # Collect promising candidates for report
            if row["auto_decision"] in ("candidate_link", "possible_link"):
                report_rows.append(row)

    return report_rows


def fmt(value):
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def analyze_links(
    ble_csv,
    btclassic_csv,
    output_dir,
    slot_us=625.0,
    search_window_us=10000.0,
    score_threshold_us=15.0,
    min_valid_pairs=5,
    min_id_occurrences=5,
    ble_aa_tolerance_bits=2,
    bt_id_tolerance_bits=0,
    outlier_window_us=0.0,
):
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ble_rows = read_csv(ble_csv)
    bt_rows = read_csv(btclassic_csv)
    ble_groups = build_ble_groups(
        ble_rows, ble_aa_tolerance_bits, min_id_occurrences
    )
    bt_key = "bdaddr" if any(row.get("bdaddr") for row in bt_rows) else "lap"
    bt_digits = 12 if bt_key == "bdaddr" else 6
    bt_groups, bt_aliases = tolerant_group_rows(bt_rows, bt_key, bt_digits, bt_id_tolerance_bits)
    bt_groups, bt_aliases = filter_groups_by_min_occurrences(
        bt_groups, bt_aliases, min_id_occurrences
    )

    pair_scores = []
    pair_offset_payloads = []
    for ble_group in ble_groups:
        ble_times = row_times_us(ble_group["rows"])
        for bt_id, bt_group in bt_groups.items():
            bt_times = row_times_us(bt_group)
            wrapped, ble_pair_times, _pairs = compute_pair_offsets(
                ble_times, bt_times, slot_us, search_window_us
            )
            score = compute_link_score(wrapped, slot_us, outlier_window_us)
            row = {
                "ble_access_address": ble_group["ble_access_address"],
                "ble_access_address_aliases": ble_group["ble_access_address_aliases"],
                "ble_device_address": ble_group["ble_device_address"],
                "ble_device_address_aliases": ble_group["ble_device_address_aliases"],
                "bt_lap": bt_id if bt_key == "lap" else "",
                "bt_bdaddr": bt_id if bt_key == "bdaddr" else "",
                "bt_id_aliases": format_aliases(bt_aliases[bt_id], bt_digits),
                "valid_pair_count": score["valid_pair_count"],
                "rmse_us": fmt(score["rmse_us"]),
                "rmse_demean_us": fmt(score["rmse_demean_us"]),
                "mode_delta_us": fmt(score["mode_delta_us"]),
                "mean_delta_us": fmt(score["mean_delta_us"]),
                "median_delta_us": fmt(score["median_delta_us"]),
                "std_delta_us": fmt(score["std_delta_us"]),
                "mad_us": fmt(score["mad_us"]),
                "iqr_us": fmt(score["iqr_us"]),
                "decision": "",
                "ambiguity_flag": False,
                "insufficient_packets_flag": score["valid_pair_count"] < min_valid_pairs,
                "filtered_pair_count": score.get("filtered_pair_count", score["valid_pair_count"]),
                "_ble_group_id": ble_group["group_id"],
                "_ble_display_id": ble_group["display_id"],
            }
            pair_scores.append(row)
            pair_offset_payloads.append((row, ble_pair_times, wrapped))

    for ble_group in ble_groups:
        rows = [row for row in pair_scores if row["_ble_group_id"] == ble_group["group_id"]]
        usable = [row for row in rows if not row["insufficient_packets_flag"] and row["rmse_demean_us"] != ""]
        usable.sort(key=lambda row: float(row["rmse_demean_us"]))
        ambiguous_ids = set()
        if len(usable) >= 2:
            if abs(float(usable[1]["rmse_demean_us"]) - float(usable[0]["rmse_demean_us"])) <= score_threshold_us:
                ambiguous_ids.update(id(row) for row in usable[:2])
        for row in rows:
            score = {
                "valid_pair_count": int(row["valid_pair_count"]),
                "rmse_demean_us": float(row["rmse_demean_us"]) if row["rmse_demean_us"] else np.inf,
            }
            row["ambiguity_flag"] = id(row) in ambiguous_ids
            row["decision"] = make_link_decision(
                score, score_threshold_us, min_valid_pairs, row["ambiguity_flag"]
            )

    link_results = []
    for ble_group in ble_groups:
        rows = [row for row in pair_scores if row["_ble_group_id"] == ble_group["group_id"]]
        usable = [row for row in rows if row["rmse_demean_us"] != ""]
        sufficient = [row for row in usable if not row["insufficient_packets_flag"]]
        candidate_pool = sufficient if sufficient else usable
        candidate_pool.sort(key=lambda row: float(row["rmse_demean_us"]))
        if not candidate_pool:
            link_results.append(
                {
                    "ble_access_address": ble_group["ble_access_address"],
                    "ble_access_address_aliases": ble_group["ble_access_address_aliases"],
                    "ble_device_address": ble_group["ble_device_address"],
                    "ble_device_address_aliases": ble_group["ble_device_address_aliases"],
                    "best_bt_lap": "",
                    "best_bt_bdaddr": "",
                    "best_bt_id_aliases": "",
                    "best_score_us": "",
                    "valid_pair_count": 0,
                    "decision": "insufficient_packets",
                    "reason": "no valid BLE/BT packet pairs in the search window",
                }
            )
            continue
        best = candidate_pool[0]
        reason = (
            f"rmse_demean_us={best['rmse_demean_us']}, "
            f"valid_pair_count={best['valid_pair_count']}, "
            f"threshold={score_threshold_us} us"
        )
        link_results.append(
            {
                "ble_access_address": ble_group["ble_access_address"],
                "ble_access_address_aliases": ble_group["ble_access_address_aliases"],
                "ble_device_address": ble_group["ble_device_address"],
                "ble_device_address_aliases": ble_group["ble_device_address_aliases"],
                "best_bt_lap": best["bt_lap"],
                "best_bt_bdaddr": best["bt_bdaddr"],
                "best_bt_id_aliases": best["bt_id_aliases"],
                "best_score_us": best["rmse_demean_us"],
                "valid_pair_count": best["valid_pair_count"],
                "decision": best["decision"],
                "reason": reason,
            }
        )

    # --- Auto-classification ---
    report_rows = classify_links(
        pair_scores,
        pair_offset_payloads,
        score_threshold_us=score_threshold_us,
        min_valid_pairs=min_valid_pairs,
        slot_us=slot_us,
        ble_group_key="_ble_group_id",
    )

    # Build classification report rows with ID info
    report_out = []
    for row in report_rows:
        ble_id = row.get("ble_device_address") or row.get("ble_access_address") or ""
        ble_id_type = "device_address" if row.get("ble_device_address") else "access_address"
        matched_id = row.get("bt_bdaddr") or row.get("bt_lap") or ""
        matched_id_type = "bdaddr" if row.get("bt_bdaddr") else "lap"
        report_out.append({
            "ble_id": ble_id,
            "ble_id_type": ble_id_type,
            "matched_id": matched_id,
            "matched_id_type": matched_id_type,
            "rmse_demean_us": row.get("rmse_demean_us", ""),
            "valid_pair_count": row.get("valid_pair_count", 0),
            "rank_in_ble_group": row.get("rank_in_ble_group", ""),
            "rmse_rel_score": row.get("rmse_rel_score", ""),
            "gap_ratio_vs_runner_up": row.get("gap_ratio_vs_runner_up", ""),
            "circular_concentration": row.get("circular_concentration", ""),
            "auto_decision": row.get("auto_decision", ""),
        })
    report_out.sort(key=lambda r: float(r.get("rmse_demean_us") or 999))

    for row in pair_scores:
        row.pop("_ble_group_id", None)
        row.pop("_ble_display_id", None)

    write_csv(output_dir / "pair_scores.csv", pair_scores, PAIR_SCORE_FIELDS)
    write_csv(output_dir / "link_results.csv", link_results, LINK_RESULT_FIELDS)
    write_csv(output_dir / "classification_report.csv", report_out, CLASSIFICATION_REPORT_FIELDS)

    # Generate scatter plots for all pairs with data
    for row, ble_times, wrapped in pair_offset_payloads:
        if len(wrapped) > 0:
            plot_offset_scatter(row, ble_times, wrapped, figures_dir)
    plot_rmse_ecdf(pair_scores, figures_dir / "rmse_ecdf.png")
    plot_best_pairs_hist(pair_scores, pair_offset_payloads, figures_dir / "best_pairs_hist.png")
    plot_all_pair_offsets(pair_offset_payloads, figures_dir / "pair_offsets_colored.png")
    return pair_scores, link_results, pair_offset_payloads


def analyze_ble_adv_conn_links(
    ble_csv,
    output_dir,
    slot_us=625.0,
    search_window_us=10000.0,
    score_threshold_us=15.0,
    min_valid_pairs=5,
    min_id_occurrences=5,
    ble_aa_tolerance_bits=2,
    outlier_window_us=0.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ble_rows = read_csv(ble_csv)
    adv_groups, conn_groups = build_ble_adv_conn_groups(
        ble_rows, ble_aa_tolerance_bits, min_id_occurrences
    )

    pair_scores = []
    pair_offset_payloads = []
    for adv_group in adv_groups:
        adv_times = row_times_us(adv_group["rows"])
        for conn_group in conn_groups:
            conn_times = row_times_us(conn_group["rows"])
            wrapped, adv_pair_times, _pairs = compute_pair_offsets(
                adv_times, conn_times, slot_us, search_window_us
            )
            score = compute_link_score(wrapped, slot_us, outlier_window_us)
            row = {
                "advertiser_address": adv_group["advertiser_address"],
                "advertiser_address_aliases": adv_group["advertiser_address_aliases"],
                "connection_access_address": conn_group["connection_access_address"],
                "connection_access_address_aliases": conn_group["connection_access_address_aliases"],
                "advertising_packet_count": len(adv_group["rows"]),
                "connection_packet_count": len(conn_group["rows"]),
                "valid_pair_count": score["valid_pair_count"],
                "rmse_us": fmt(score["rmse_us"]),
                "rmse_demean_us": fmt(score["rmse_demean_us"]),
                "mode_delta_us": fmt(score["mode_delta_us"]),
                "mean_delta_us": fmt(score["mean_delta_us"]),
                "median_delta_us": fmt(score["median_delta_us"]),
                "std_delta_us": fmt(score["std_delta_us"]),
                "mad_us": fmt(score["mad_us"]),
                "iqr_us": fmt(score["iqr_us"]),
                "decision": "",
                "ambiguity_flag": False,
                "insufficient_packets_flag": score["valid_pair_count"] < min_valid_pairs,
                "filtered_pair_count": score.get("filtered_pair_count", score["valid_pair_count"]),
            }
            pair_scores.append(row)
            pair_offset_payloads.append((row, adv_pair_times, wrapped))

    for adv_group in adv_groups:
        rows = [row for row in pair_scores if row["advertiser_address"] == adv_group["advertiser_address"]]
        usable = [row for row in rows if not row["insufficient_packets_flag"] and row["rmse_demean_us"] != ""]
        usable.sort(key=lambda row: float(row["rmse_demean_us"]))
        ambiguous_ids = set()
        if len(usable) >= 2:
            if abs(float(usable[1]["rmse_demean_us"]) - float(usable[0]["rmse_demean_us"])) <= score_threshold_us:
                ambiguous_ids.update(id(row) for row in usable[:2])
        for row in rows:
            score = {
                "valid_pair_count": int(row["valid_pair_count"]),
                "rmse_demean_us": float(row["rmse_demean_us"]) if row["rmse_demean_us"] else np.inf,
            }
            row["ambiguity_flag"] = id(row) in ambiguous_ids
            row["decision"] = make_link_decision(
                score, score_threshold_us, min_valid_pairs, row["ambiguity_flag"]
            )

    link_results = []
    for adv_group in adv_groups:
        rows = [row for row in pair_scores if row["advertiser_address"] == adv_group["advertiser_address"]]
        usable = [row for row in rows if row["rmse_demean_us"] != ""]
        sufficient = [row for row in usable if not row["insufficient_packets_flag"]]
        candidate_pool = sufficient if sufficient else usable
        candidate_pool.sort(key=lambda row: float(row["rmse_demean_us"]))
        if not candidate_pool:
            link_results.append(
                {
                    "advertiser_address": adv_group["advertiser_address"],
                    "advertiser_address_aliases": adv_group["advertiser_address_aliases"],
                    "best_connection_access_address": "",
                    "best_connection_access_address_aliases": "",
                    "best_score_us": "",
                    "valid_pair_count": 0,
                    "decision": "insufficient_packets",
                    "reason": "no valid advertising/connection BLE packet pairs in the search window",
                }
            )
            continue

        best = candidate_pool[0]
        link_results.append(
            {
                "advertiser_address": adv_group["advertiser_address"],
                "advertiser_address_aliases": adv_group["advertiser_address_aliases"],
                "best_connection_access_address": best["connection_access_address"],
                "best_connection_access_address_aliases": best["connection_access_address_aliases"],
                "best_score_us": best["rmse_demean_us"],
                "valid_pair_count": best["valid_pair_count"],
                "decision": best["decision"],
                "reason": (
                    f"rmse_demean_us={best['rmse_demean_us']}, "
                    f"valid_pair_count={best['valid_pair_count']}, "
                    f"threshold={score_threshold_us} us"
                ),
            }
        )

    # --- Auto-classification ---
    # Add temporary group key for classify_links
    for row in pair_scores:
        row["_adv_group_id"] = row.get("advertiser_address", "")

    report_rows = classify_links(
        pair_scores,
        pair_offset_payloads,
        score_threshold_us=score_threshold_us,
        min_valid_pairs=min_valid_pairs,
        slot_us=slot_us,
        ble_group_key="_adv_group_id",
    )

    # Build classification report rows
    report_out = []
    for row in report_rows:
        report_out.append({
            "ble_id": row.get("advertiser_address", ""),
            "ble_id_type": "advertiser_address",
            "matched_id": row.get("connection_access_address", ""),
            "matched_id_type": "connection_access_address",
            "rmse_demean_us": row.get("rmse_demean_us", ""),
            "valid_pair_count": row.get("valid_pair_count", 0),
            "rank_in_ble_group": row.get("rank_in_ble_group", ""),
            "rmse_rel_score": row.get("rmse_rel_score", ""),
            "gap_ratio_vs_runner_up": row.get("gap_ratio_vs_runner_up", ""),
            "circular_concentration": row.get("circular_concentration", ""),
            "auto_decision": row.get("auto_decision", ""),
        })
    report_out.sort(key=lambda r: float(r.get("rmse_demean_us") or 999))

    # Clean up temporary key
    for row in pair_scores:
        row.pop("_adv_group_id", None)

    write_csv(output_dir / "ble_adv_conn_pair_scores.csv", pair_scores, BLE_BLE_PAIR_SCORE_FIELDS)
    write_csv(output_dir / "ble_adv_conn_link_results.csv", link_results, BLE_BLE_LINK_RESULT_FIELDS)
    write_csv(output_dir / "classification_report.csv", report_out, CLASSIFICATION_REPORT_FIELDS)

    # Generate scatter plots for all pairs with data
    for row, adv_times, wrapped in pair_offset_payloads:
        if len(wrapped) > 0:
            plot_ble_ble_offset_scatter(row, adv_times, wrapped, figures_dir)
    plot_ble_ble_rmse_ecdf(pair_scores, figures_dir / "ble_adv_conn_rmse_ecdf.png")
    plot_ble_ble_best_pairs_hist(
        pair_scores, pair_offset_payloads, figures_dir / "ble_adv_conn_best_pairs_hist.png"
    )
    plot_ble_ble_all_pair_offsets(
        pair_offset_payloads, figures_dir / "ble_adv_conn_pair_offsets_colored.png"
    )
    return pair_scores, link_results, pair_offset_payloads


class _UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _classify_id_type(raw_id):
    raw = raw_id.strip().upper().replace("0X", "")
    if len(raw) == 6:
        return "bt_lap"
    if len(raw) == 8:
        return "ble_access_address"
    if len(raw) == 12:
        return "ble_device_address"
    return "unknown"


def build_device_groups(
    ble_csv,
    link_dir,
    ble_link_dir,
):
    """Build transitive device groups from link and ble-link analysis results.

    Edges come from three sources:
    1. ADV packets in ble_packets.csv: access_address <-> ble_device_address
    2. link classification_report: ble_access_address <-> bt_lap
    3. ble-link classification_report: advertiser_address <-> connection_access_address

    Returns list of device group dicts.
    """
    link_dir = Path(link_dir)
    ble_link_dir = Path(ble_link_dir)

    uf = _UnionFind()
    edge_records = []

    # Source 1: ADV packets provide access_address <-> device_address edges
    # But filter out the BLE standard advertising access address
    # (spec 0x8E89BED6, captured as D6BE898E in little-endian byte order)
    # since all BLE devices share it — it cannot identify a specific device.
    BLE_ADV_AA = normalize_hex_id("D6BE898E", 8)  # standard BLE advertising AA in capture byte order
    ble_rows = read_csv(ble_csv)
    adv_aa_to_dev = {}
    for row in ble_rows:
        dev_addr = row.get("ble_device_address", "").strip()
        aa = row.get("access_address", "").strip()
        if dev_addr and aa:
            aa_norm = normalize_hex_id(aa, 8)
            dev_norm = normalize_hex_id(dev_addr, 12)
            if aa_norm and dev_norm and hamming_distance_bits(aa_norm, BLE_ADV_AA) > 3:
                key = (aa_norm, dev_norm)
                adv_aa_to_dev[key] = adv_aa_to_dev.get(key, 0) + 1
    for (aa, dev), count in adv_aa_to_dev.items():
        aa_label = f"0x{aa}"
        uf.union(aa_label, dev)
        edge_records.append({
            "from_id": aa_label,
            "to_id": dev,
            "from_type": "ble_access_address",
            "to_type": "ble_device_address",
            "edge_source": "adv_packet",
            "rmse_us": "",
            "valid_pair_count": count,
        })

    # Source 2: link classification_report: ble_access_address <-> bt_lap
    link_report_path = link_dir / "classification_report.csv"
    if link_report_path.exists():
        for row in read_csv(link_report_path):
            if row.get("auto_decision", "") not in ("candidate_link", "possible_link"):
                continue
            ble_id = row.get("ble_id", "").strip()
            matched_id = row.get("matched_id", "").strip()
            rmse = row.get("rmse_demean_us", "")
            pairs = row.get("valid_pair_count", "0")
            if ble_id and matched_id:
                uf.union(ble_id, matched_id)
                edge_records.append({
                    "from_id": ble_id,
                    "to_id": matched_id,
                    "from_type": row.get("ble_id_type", ""),
                    "to_type": row.get("matched_id_type", ""),
                    "edge_source": "link",
                    "rmse_us": rmse,
                    "valid_pair_count": pairs,
                })

    # Source 3: ble-link classification_report: advertiser_address <-> connection_access_address
    ble_link_report_path = ble_link_dir / "classification_report.csv"
    if ble_link_report_path.exists():
        for row in read_csv(ble_link_report_path):
            if row.get("auto_decision", "") not in ("candidate_link", "possible_link"):
                continue
            ble_id = row.get("ble_id", "").strip()
            matched_id = row.get("matched_id", "").strip()
            rmse = row.get("rmse_demean_us", "")
            pairs = row.get("valid_pair_count", "0")
            if ble_id and matched_id:
                uf.union(ble_id, matched_id)
                edge_records.append({
                    "from_id": ble_id,
                    "to_id": matched_id,
                    "from_type": row.get("ble_id_type", ""),
                    "to_type": row.get("matched_id_type", ""),
                    "edge_source": "ble_link",
                    "rmse_us": rmse,
                    "valid_pair_count": pairs,
                })

    # Extract connected components
    components = defaultdict(set)
    all_ids = set()
    for edge in edge_records:
        all_ids.add(edge["from_id"])
        all_ids.add(edge["to_id"])
    for node in all_ids:
        root = uf.find(node)
        components[root].add(node)

    # Build group records
    groups = []
    for idx, (root, members) in enumerate(sorted(components.items(), key=lambda x: -len(x[1]))):
        bt_laps = sorted(m for m in members if _classify_id_type(m) == "bt_lap")
        ble_aas = sorted(m for m in members if _classify_id_type(m) == "ble_access_address")
        ble_devs = sorted(m for m in members if _classify_id_type(m) == "ble_device_address")

        group_edges = []
        best_rmse = ""
        for edge in edge_records:
            if edge["from_id"] in members or edge["to_id"] in members:
                group_edges.append(edge)
                rmse_val = edge.get("rmse_us", "")
                if rmse_val:
                    try:
                        f = float(rmse_val)
                        if best_rmse == "" or f < float(best_rmse):
                            best_rmse = rmse_val
                    except (ValueError, TypeError):
                        pass

        edge_desc_parts = []
        for e in group_edges:
            edge_desc_parts.append(f"{e['from_id']}--{e['to_id']}({e['edge_source']})")

        groups.append({
            "group_id": f"device_{idx}",
            "identifiers": ";".join(sorted(members)),
            "bt_lap_list": ";".join(bt_laps),
            "ble_access_address_list": ";".join(ble_aas),
            "ble_device_address_list": ";".join(ble_devs),
            "link_edges": ";".join(edge_desc_parts),
            "best_rmse_us": best_rmse,
            "edge_count": len(group_edges),
        })

    return groups
