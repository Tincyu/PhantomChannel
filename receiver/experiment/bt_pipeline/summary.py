from pathlib import Path


def unique_count(rows, key):
    return len({row.get(key, "") for row in rows if row.get(key, "")})


def ble_label(row):
    return row.get("ble_device_address") or row.get("ble_access_address") or "N/A"


def generate_summary_md(
    output_path,
    input_file,
    sample_rate,
    duration,
    ble_packets,
    bt_packets,
    pair_scores,
    link_results,
    slot_us=625.0,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed = any(row.get("decision", "").startswith("candidate") for row in link_results)
    insufficient = [row for row in pair_scores if str(row.get("insufficient_packets_flag")) == "True"]
    sample_period_us = 1e6 / float(sample_rate)
    slot_samples = float(sample_rate) * slot_us / 1e6

    lines = [
        "# Bluetooth Clock-Link Experiment Summary",
        "",
        f"- Input file: `{input_file}`",
        f"- Sample rate: {float(sample_rate):.0f} Hz",
        f"- Capture duration: {duration}",
        f"- Timing: 1 sample = {sample_period_us:g} us; {slot_us:g} us = {slot_samples:g} samples.",
        f"- BLE packet count: {len(ble_packets)}",
        f"- Classic Bluetooth packet count: {len(bt_packets)}",
        f"- BLE Access Address count: {unique_count(ble_packets, 'access_address')}",
        f"- BLE advertiser/device-address count: {unique_count(ble_packets, 'ble_device_address')}",
        f"- Classic Bluetooth LAP/BDADDR count: {unique_count(bt_packets, 'lap') or unique_count(bt_packets, 'bdaddr')}",
        f"- Candidate pair count: {len(pair_scores)}",
        f"- Insufficient-packet pair count: {len(insufficient)}",
        f"- Observed {slot_us:g} us alignment: {'yes' if observed else 'not confirmed'}",
        "",
        "## Candidate Results",
        "",
    ]
    if link_results:
        for row in link_results:
            lines.append(
                f"- BLE {ble_label(row)} -> BT "
                f"{row.get('best_bt_lap') or row.get('best_bt_bdaddr') or 'N/A'}: "
                f"{row.get('decision')} ({row.get('reason')})"
            )
    else:
        lines.append("- No candidate result rows were generated.")

    # Auto-classification results
    candidate_links = [row for row in pair_scores if row.get("auto_decision") == "candidate_link"]
    possible_links = [row for row in pair_scores if row.get("auto_decision") == "possible_link"]
    no_link_count = sum(1 for row in pair_scores if row.get("auto_decision") == "no_link")
    insuff_count = sum(1 for row in pair_scores if row.get("auto_decision") == "insufficient_packets")

    lines.extend([
        "",
        "## Auto-Classification",
        "",
        f"- candidate_link: {len(candidate_links)}",
        f"- possible_link: {len(possible_links)}",
        f"- no_link: {no_link_count}",
        f"- insufficient_packets: {insuff_count}",
        "",
    ])

    if candidate_links:
        lines.append("### Candidate Links (absolute threshold met)")
        lines.append("")
        for row in sorted(candidate_links, key=lambda r: float(r.get("rmse_demean_us") or 999)):
            ble_id = row.get("ble_device_address") or row.get("ble_access_address") or "N/A"
            bt_id = row.get("bt_bdaddr") or row.get("bt_lap") or "N/A"
            lines.append(
                f"- BLE {ble_id} -> BT {bt_id}: "
                f"rmse={row.get('rmse_demean_us')} us, "
                f"pairs={row.get('valid_pair_count')}, "
                f"conc={row.get('circular_concentration', '')}"
            )
        lines.append("")

    if possible_links:
        lines.append("### Possible Links (relative ranking)")
        lines.append("")
        for row in sorted(possible_links, key=lambda r: float(r.get("rmse_demean_us") or 999)):
            ble_id = row.get("ble_device_address") or row.get("ble_access_address") or "N/A"
            bt_id = row.get("bt_bdaddr") or row.get("bt_lap") or "N/A"
            lines.append(
                f"- BLE {ble_id} -> BT {bt_id}: "
                f"rmse={row.get('rmse_demean_us')} us, "
                f"pairs={row.get('valid_pair_count')}, "
                f"rel_score={row.get('rmse_rel_score', '')}, "
                f"gap_ratio={row.get('gap_ratio_vs_runner_up', '')}, "
                f"conc={row.get('circular_concentration', '')}"
            )
        lines.append("")

    if not candidate_links and not possible_links:
        lines.append("- No promising candidates found by auto-classification.")
        lines.append("")

    lines.extend(
        [
            "",
            "## Single-Channel Limitations",
            "",
            "- Single-channel capture only observes BLE packets that hop into the captured channel or bandwidth.",
            "- Classic Bluetooth uses 79 1 MHz hop channels, so this capture also sees only a subset of packets.",
            "- Low packet counts make RMSE unstable; insufficient packet counts are reported separately from no-link decisions.",
            "- A 20 s whole-capture analysis is generally more reliable than a short 1 s window.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_ble_link_summary_md(
    output_path,
    input_file,
    sample_rate,
    duration,
    ble_packets,
    pair_scores,
    link_results,
    slot_us=625.0,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed = any(row.get("decision", "").startswith("candidate") for row in link_results)
    insufficient = [row for row in pair_scores if str(row.get("insufficient_packets_flag")) == "True"]
    adv_packets = [row for row in ble_packets if row.get("ble_device_address")]
    conn_packets = [row for row in ble_packets if row.get("access_address") and not row.get("ble_device_address")]
    sample_period_us = 1e6 / float(sample_rate)
    slot_samples = float(sample_rate) * slot_us / 1e6

    lines = [
        "# BLE Advertising/Connection Clock-Link Summary",
        "",
        f"- Input file: `{input_file}`",
        f"- Sample rate: {float(sample_rate):.0f} Hz",
        f"- Capture duration: {duration}",
        f"- Timing: 1 sample = {sample_period_us:g} us; {slot_us:g} us = {slot_samples:g} samples.",
        f"- BLE packet count: {len(ble_packets)}",
        f"- Advertising packet count: {len(adv_packets)}",
        f"- Connection packet count: {len(conn_packets)}",
        f"- Advertising device-address count: {unique_count(ble_packets, 'ble_device_address')}",
        f"- Connection access-address count: {unique_count(conn_packets, 'access_address')}",
        f"- Candidate pair count: {len(pair_scores)}",
        f"- Insufficient-packet pair count: {len(insufficient)}",
        f"- Observed {slot_us:g} us alignment: {'yes' if observed else 'not confirmed'}",
        "",
        "## Candidate Results",
        "",
    ]
    if link_results:
        for row in link_results:
            lines.append(
                f"- ADV {row.get('advertiser_address') or 'N/A'} -> CONN "
                f"{row.get('best_connection_access_address') or 'N/A'}: "
                f"{row.get('decision')} ({row.get('reason')})"
            )
    else:
        lines.append("- No candidate result rows were generated.")

    # Auto-classification results
    candidate_links = [row for row in pair_scores if row.get("auto_decision") == "candidate_link"]
    possible_links = [row for row in pair_scores if row.get("auto_decision") == "possible_link"]
    no_link_count = sum(1 for row in pair_scores if row.get("auto_decision") == "no_link")
    insuff_count = sum(1 for row in pair_scores if row.get("auto_decision") == "insufficient_packets")

    lines.extend([
        "",
        "## Auto-Classification",
        "",
        f"- candidate_link: {len(candidate_links)}",
        f"- possible_link: {len(possible_links)}",
        f"- no_link: {no_link_count}",
        f"- insufficient_packets: {insuff_count}",
        "",
    ])

    if candidate_links:
        lines.append("### Candidate Links (absolute threshold met)")
        lines.append("")
        for row in sorted(candidate_links, key=lambda r: float(r.get("rmse_demean_us") or 999)):
            lines.append(
                f"- ADV {row.get('advertiser_address') or 'N/A'} -> CONN "
                f"{row.get('connection_access_address') or 'N/A'}: "
                f"rmse={row.get('rmse_demean_us')} us, "
                f"pairs={row.get('valid_pair_count')}, "
                f"conc={row.get('circular_concentration', '')}"
            )
        lines.append("")

    if possible_links:
        lines.append("### Possible Links (relative ranking)")
        lines.append("")
        for row in sorted(possible_links, key=lambda r: float(r.get("rmse_demean_us") or 999)):
            lines.append(
                f"- ADV {row.get('advertiser_address') or 'N/A'} -> CONN "
                f"{row.get('connection_access_address') or 'N/A'}: "
                f"rmse={row.get('rmse_demean_us')} us, "
                f"pairs={row.get('valid_pair_count')}, "
                f"rel_score={row.get('rmse_rel_score', '')}, "
                f"gap_ratio={row.get('gap_ratio_vs_runner_up', '')}, "
                f"conc={row.get('circular_concentration', '')}"
            )
        lines.append("")

    if not candidate_links and not possible_links:
        lines.append("- No promising candidates found by auto-classification.")
        lines.append("")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- This analysis looks for stable {slot_us:g} us timing relationships between BLE advertising traffic and BLE connection traffic from the same capture.",
            "- A low RMSE after mode subtraction suggests the two BLE streams may share a common clock base.",
            "- Advertising and connection traffic can be sparse or bursty, so low packet counts still need caution.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_device_group_summary_md(
    output_path,
    input_dir,
    device_groups,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    multi_id_groups = [g for g in device_groups if g["edge_count"] > 0]
    groups_with_bt = [g for g in device_groups if g.get("bt_lap_list")]
    groups_with_conn = [g for g in device_groups if g.get("ble_access_address_list")]
    groups_with_adv = [g for g in device_groups if g.get("ble_device_address_list")]

    lines = [
        "# Device Grouping Summary",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Total device groups: {len(device_groups)}",
        f"- Groups with linked edges: {len(multi_id_groups)}",
        f"- Groups containing BT LAP: {len(groups_with_bt)}",
        f"- Groups containing BLE connection AA: {len(groups_with_conn)}",
        f"- Groups containing BLE advertising address: {len(groups_with_adv)}",
        "",
        "## Device Groups",
        "",
    ]

    for g in device_groups:
        if g["edge_count"] == 0:
            continue
        lines.append(f"### {g['group_id']}")
        lines.append("")
        if g.get("bt_lap_list"):
            lines.append(f"- **BT LAP**: {g['bt_lap_list']}")
        if g.get("ble_access_address_list"):
            lines.append(f"- **BLE Connection AA**: {g['ble_access_address_list']}")
        if g.get("ble_device_address_list"):
            lines.append(f"- **BLE Advertising Addr**: {g['ble_device_address_list']}")
        if g.get("best_rmse_us"):
            lines.append(f"- **Best RMSE**: {g['best_rmse_us']} us")
        lines.append(f"- **Edge count**: {g['edge_count']}")
        if g.get("link_edges"):
            lines.append(f"- **Links**: {g['link_edges']}")
        lines.append("")

    if not multi_id_groups:
        lines.append("- No multi-identifier device groups found.")
        lines.append("")

    lines.extend([
        "",
        "## Notes",
        "",
        "- Device groups are built by transitive closure of pairwise link analysis results.",
        "- Edges come from: ADV packet data (AA<->device_addr), link analysis (AA<->LAP), ble-link analysis (ADV<->CONN).",
        "- Only candidate_link and possible_link edges are used for grouping.",
        "- A group with BT LAP + BLE connection AA + BLE advertising address indicates a dual-mode device.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
