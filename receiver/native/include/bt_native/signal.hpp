#pragma once

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace bt_native {

struct SegmentSummary {
    std::int64_t offset;
    std::int64_t length;
    double power_sum;
};

struct BlePacketCandidate {
    std::string access_address;
    int payload_len;
    double score;
    std::string ble_pdu_type;
    std::string whitened_pdu_hex;
    std::string dewhitened_pdu_hex;
    std::string captured_crc_hex;
    std::string post_crc_hex;
    std::string crc_and_post_crc_hex;
    std::string crc_capture_status;
    std::string advertiser_address;
    std::string advertiser_address_type;
    std::string peer_address;
    std::string peer_address_type;
    std::string ble_device_address;
    std::int64_t sample_offset;
    std::int64_t sample_index;
    std::int64_t segment_index;
    double rssi;
};

struct BrAccessCodeMatch {
    std::int64_t offset;
    std::uint32_t lap;
    bool valid;
};

struct BrHeaderCandidate {
    int uap;
    int clk;
    std::uint32_t header;
    int lfsr;
    std::int64_t payload_bit_count;
    std::uint16_t payload_prefix_bits;
    int payload_prefix_count;
};

struct BrPacketDecision {
    std::string status;
    int uap;
    int clk;
    std::string type_name;
    int type_val;
    int length;
    int total_bytes;
    std::optional<int> locked_uap;
    int miss_count;
    bool hec_ok;
};

struct BrPacketCandidate {
    std::int64_t sample_index;
    std::int64_t segment_index;
    std::uint32_t lap;
    double rssi;
    std::optional<double> cfo_hz;
    std::int64_t raw_bits_len;
    std::vector<BrHeaderCandidate> header_candidates;
    std::vector<std::uint8_t> payload_bits;
};

std::vector<double> gfsk_demodulate(
    const std::complex<float>* samples,
    std::int64_t sample_count,
    double gain);

std::vector<std::uint8_t> decision_bits(
    const double* freq_dev,
    std::int64_t sample_count,
    std::int64_t samples_per_symbol);

std::optional<std::string> detect_ble_access_address(
    const std::uint8_t* bits,
    std::int64_t bit_count,
    const std::string& ble_mode);

bool valid_ble_connection_access_address(const std::string& access_address_raw);

std::optional<BlePacketCandidate> parse_ble_packet_bits(
    const std::uint8_t* bits,
    std::int64_t bit_count,
    const std::string& ble_mode,
    std::int64_t iq_len,
    int channel);

std::vector<BlePacketCandidate> parse_ble_segments(
    const std::complex<float>* samples,
    std::int64_t sample_count,
    const std::int64_t* offsets,
    const std::int64_t* lengths,
    const std::int64_t* score_lengths,
    const std::int64_t* sample_indices,
    std::int64_t segment_count,
    double gain,
    double score_threshold,
    int channel,
    int thread_count = 1);

std::uint64_t build_bluetooth_sync_word(std::uint32_t lap);

BrAccessCodeMatch find_br_access_code(
    const std::uint8_t* bits,
    std::int64_t bit_count);

std::vector<BrHeaderCandidate> decode_br_header_candidates(
    const std::uint8_t* raw_bits,
    std::int64_t bit_count,
    const int* candidate_uaps,
    std::int64_t candidate_uap_count,
    bool stop_after_first);

std::vector<BrPacketCandidate> build_br_packet_candidates(
    const std::complex<float>* samples,
    std::int64_t sample_count,
    const std::int64_t* offsets,
    const std::int64_t* lengths,
    const std::int64_t* sample_indices,
    const std::int64_t* segment_indices,
    std::int64_t segment_count,
    const double* lpf_taps,
    std::int64_t tap_count,
    double gain,
    int samples_per_bit,
    double sample_rate,
    int thread_count = 1);

BrPacketDecision process_br_header_candidates(
    const std::vector<BrHeaderCandidate>& candidates,
    std::int64_t raw_bits_len,
    std::optional<int> locked_uap,
    int miss_count,
    int max_miss,
    bool retry_after_unlock);

std::vector<BrPacketDecision> process_br_header_candidate_sequence(
    const std::vector<std::vector<BrHeaderCandidate>>& candidate_batches,
    const std::vector<std::int64_t>& raw_bits_lens,
    std::optional<int> locked_uap,
    int miss_count,
    int max_miss,
    bool retry_after_unlock);

double compute_rssi_db(
    const std::complex<float>* samples,
    std::int64_t sample_count,
    double gain_offset_db);

std::vector<SegmentSummary> summarize_segments(
    const std::complex<float>* samples,
    std::int64_t sample_count,
    const std::int64_t* offsets,
    const std::int64_t* lengths,
    std::int64_t segment_count);

}  // namespace bt_native
