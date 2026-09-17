#include "bt_native/signal.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <thread>
#include <sstream>
#include <stdexcept>

namespace bt_native {

std::vector<double> gfsk_demodulate(
    const std::complex<float>* samples,
    std::int64_t sample_count,
    double gain)
{
    if (sample_count <= 1) {
        return {};
    }

    std::vector<double> demod(static_cast<std::size_t>(sample_count - 1));
    for (std::int64_t idx = 1; idx < sample_count; ++idx) {
        const std::complex<double> prev(samples[idx - 1].real(), samples[idx - 1].imag());
        const std::complex<double> curr(samples[idx].real(), samples[idx].imag());
        const double dphase = std::arg(curr * std::conj(prev));
        demod[static_cast<std::size_t>(idx - 1)] = gain * dphase;
    }
    return demod;
}

std::vector<std::uint8_t> decision_bits(
    const double* freq_dev,
    std::int64_t sample_count,
    std::int64_t samples_per_symbol)
{
    if (samples_per_symbol <= 0) {
        throw std::invalid_argument("samples_per_symbol must be positive");
    }

    std::vector<std::uint8_t> bits;
    for (std::int64_t idx = samples_per_symbol / 2; idx < sample_count; idx += samples_per_symbol) {
        bits.push_back(freq_dev[idx] > 0.0 ? 1 : 0);
    }
    return bits;
}

namespace {

std::uint64_t compute_remainder(std::uint64_t value, std::uint64_t polynomial)
{
    auto bit_length = [](std::uint64_t item) {
        int length = 0;
        while (item) {
            ++length;
            item >>= 1U;
        }
        return length;
    };

    int polynomial_length = bit_length(polynomial);
    int remainder_length = bit_length(value);
    while (remainder_length >= polynomial_length && value != 0) {
        value ^= polynomial << (remainder_length - polynomial_length);
        remainder_length = bit_length(value);
    }
    return value;
}

std::uint64_t bits_to_uint_lsb(const std::uint8_t* bits, std::int64_t count)
{
    std::uint64_t value = 0;
    for (std::int64_t idx = 0; idx < count; ++idx) {
        value |= static_cast<std::uint64_t>(bits[idx] ? 1U : 0U) << idx;
    }
    return value;
}

int majority_vote_fec(const std::uint8_t* raw_header_bits)
{
    int header = 0;
    for (int idx = 0; idx < 54; idx += 3) {
        const int sum = raw_header_bits[idx] + raw_header_bits[idx + 1] + raw_header_bits[idx + 2];
        if (sum >= 2) {
            header |= 1 << (idx / 3);
        }
    }
    return header;
}

bool check_hec(int header_dewhitened, int uap)
{
    int lfsr = uap;
    for (int idx = 0; idx < 10; ++idx) {
        const int data_in = (header_dewhitened >> idx) & 0x1;
        const int lfsr_out = (lfsr >> 7) & 0x1;
        const int lfsr_in = lfsr_out ^ data_in;
        const int lfsr_adder =
            (lfsr_in << 7) | (lfsr_in << 5) | (lfsr_in << 2) | (lfsr_in << 1) | lfsr_in;
        lfsr = (lfsr << 1) & 0xff;
        lfsr ^= lfsr_adder;
    }
    for (int idx = 0; idx < 8; ++idx) {
        const int bit_rx = (header_dewhitened >> (10 + idx)) & 0x1;
        const int bit_tx = (lfsr >> (7 - idx)) & 0x1;
        if (bit_rx != bit_tx) {
            return false;
        }
    }
    return true;
}

std::pair<std::string, int> get_packet_type_info(int type_val)
{
    switch (type_val) {
    case 0:
        return {"NULL", 0};
    case 1:
        return {"POLL", 0};
    case 2:
        return {"FHS", 0};
    case 3:
        return {"DM1", 1};
    case 4:
        return {"DH1", 1};
    case 5:
        return {"HV1_or_reserved_acl", -1};
    case 6:
        return {"HV2_or_reserved_acl", -1};
    case 7:
        return {"HV3_or_reserved_acl", -1};
    case 8:
        return {"DV", 1};
    case 9:
        return {"AUX1", 1};
    case 10:
        return {"DM3", 2};
    case 11:
        return {"DH3", 2};
    case 12:
        return {"reserved_or_EV4", -1};
    case 13:
        return {"reserved_or_EV5", -1};
    case 14:
        return {"DM5", 2};
    case 15:
        return {"DH5", 2};
    default:
        return {"Unknown_" + std::to_string(type_val), 0};
    }
}

int ceil_div_int(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

int fec23_air_bits(int info_bits)
{
    return ceil_div_int(info_bits * 3, 2);
}

int compute_br_air_total_bytes(int type_val, int payload_len)
{
    constexpr int access_code_bits = 72;
    constexpr int header_raw_bits = 54;
    int payload_bits = -1;

    if (type_val == 0 || type_val == 1) {
        payload_bits = 0;
    } else if (type_val == 2) {
        payload_bits = 240;
    } else if (type_val == 3 && 0 <= payload_len && payload_len <= 17) {
        payload_bits = fec23_air_bits((1 + payload_len + 2) * 8);
    } else if (type_val == 4 && 0 <= payload_len && payload_len <= 27) {
        payload_bits = (1 + payload_len + 2) * 8;
    } else if (type_val == 5 || type_val == 6 || type_val == 7) {
        payload_bits = 240;
    } else if (type_val == 8 && 0 <= payload_len && payload_len <= 9) {
        payload_bits = 80 + fec23_air_bits((1 + payload_len + 2) * 8);
    } else if (type_val == 9 && 0 <= payload_len && payload_len <= 29) {
        payload_bits = (1 + payload_len) * 8;
    } else if (type_val == 10 && 0 <= payload_len && payload_len <= 121) {
        payload_bits = fec23_air_bits((2 + payload_len + 2) * 8);
    } else if (type_val == 11 && 0 <= payload_len && payload_len <= 183) {
        payload_bits = (2 + payload_len + 2) * 8;
    } else if (type_val == 14 && 0 <= payload_len && payload_len <= 224) {
        payload_bits = fec23_air_bits((2 + payload_len + 2) * 8);
    } else if (type_val == 15 && 0 <= payload_len && payload_len <= 339) {
        payload_bits = (2 + payload_len + 2) * 8;
    }

    if (payload_bits < 0) {
        return -1;
    }
    return ceil_div_int(access_code_bits + header_raw_bits + payload_bits, 8);
}

std::uint16_t dewhiten_payload_prefix(
    const std::uint8_t* payload_bits,
    std::int64_t payload_bit_count,
    int lfsr,
    int* prefix_count)
{
    const int count = static_cast<int>(std::min<std::int64_t>(16, payload_bit_count));
    std::uint16_t prefix = 0;
    for (int idx = 0; idx < count; ++idx) {
        const int w_out = (lfsr >> 6) & 0x1;
        lfsr = ((lfsr << 1) & 0x7f) ^ (w_out | (w_out << 4));
        prefix |= static_cast<std::uint16_t>((payload_bits[idx] ^ w_out) & 0x1) << idx;
    }
    *prefix_count = count;
    return prefix;
}

char hex_digit(std::uint8_t value)
{
    static constexpr char digits[] = "0123456789abcdef";
    return digits[value & 0x0f];
}

std::string bits_to_hex_buffer(const std::uint8_t* bits, std::int64_t bit_count)
{
    std::string output;
    output.reserve(static_cast<std::size_t>((bit_count + 3) / 4));
    for (std::int64_t idx = 0; idx < bit_count; idx += 4) {
        std::uint8_t value = 0;
        for (std::int64_t bit = 0; bit < 4; ++bit) {
            const std::int64_t source_idx = idx + bit;
            const std::uint8_t source_bit = source_idx < bit_count ? bits[source_idx] : 0;
            value |= static_cast<std::uint8_t>((source_bit ? 1 : 0) << bit);
        }
        output.push_back(hex_digit(value));
    }
    return output;
}

int hex_value(char value)
{
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return 0;
}

int hamming_distance_hex(const std::string& lhs, const std::string& rhs)
{
    int distance = 0;
    const std::size_t count = std::min(lhs.size(), rhs.size());
    for (std::size_t idx = 0; idx < count; ++idx) {
        unsigned value = static_cast<unsigned>(hex_value(lhs[idx]) ^ hex_value(rhs[idx]));
        while (value) {
            distance += static_cast<int>(value & 1U);
            value >>= 1U;
        }
    }
    distance += static_cast<int>(4 * (lhs.size() > rhs.size() ? lhs.size() - rhs.size() : rhs.size() - lhs.size()));
    return distance;
}

std::int64_t fuzzy_find(const std::string& text, const std::string& pattern, int tolerance)
{
    if (text.size() < pattern.size()) {
        return -1;
    }
    for (std::size_t idx = 0; idx <= text.size() - pattern.size(); ++idx) {
        if (hamming_distance_hex(text.substr(idx, pattern.size()), pattern) <= tolerance) {
            return static_cast<std::int64_t>(idx);
        }
    }
    return -1;
}

std::pair<std::int64_t, std::string> find_patterns(const std::string& text, const std::string& ble_mode)
{
    const std::vector<std::string> high_priority = {"aaaa", "5555"};
    const std::vector<std::string> low_priority = {"aa", "55"};
    const auto& patterns = ble_mode == "2M" ? high_priority : low_priority;
    const std::string detected_mode = ble_mode == "2M" ? "2M" : "1M";

    if (ble_mode != "1M" && ble_mode != "2M") {
        return {-1, "0M"};
    }

    for (const auto& pattern : patterns) {
        const auto pos = text.find(pattern);
        if (pos != std::string::npos) {
            return {static_cast<std::int64_t>(pos), detected_mode};
        }
    }
    for (const auto& pattern : patterns) {
        const auto pos = fuzzy_find(text, pattern, 1);
        if (pos != -1) {
            return {pos, detected_mode};
        }
    }
    return {-1, "0M"};
}

std::vector<std::int64_t> find_all_pattern_positions(
    const std::string& text,
    const std::string& ble_mode)
{
    const std::vector<std::string> high_priority = {"aaaa", "5555"};
    const std::vector<std::string> low_priority = {"aa", "55"};
    const auto& patterns = ble_mode == "2M" ? high_priority : low_priority;
    if (ble_mode != "1M" && ble_mode != "2M") {
        return {};
    }

    std::vector<std::int64_t> positions;
    for (const auto& pattern : patterns) {
        std::size_t start = 0;
        while (start + pattern.size() <= text.size()) {
            const auto pos = text.find(pattern, start);
            if (pos == std::string::npos) {
                break;
            }
            positions.push_back(static_cast<std::int64_t>(pos));
            start = pos + 1;
        }
    }
    if (positions.empty()) {
        for (const auto& pattern : patterns) {
            if (text.size() < pattern.size()) {
                continue;
            }
            for (std::size_t pos = 0; pos <= text.size() - pattern.size(); ++pos) {
                if (hamming_distance_hex(text.substr(pos, pattern.size()), pattern) <= 1) {
                    positions.push_back(static_cast<std::int64_t>(pos));
                }
            }
        }
    }
    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    return positions;
}

std::vector<std::uint8_t> reverse_bytes_to_ints(const std::string& text)
{
    std::vector<std::uint8_t> output;
    output.reserve((text.size() + 1) / 2);
    for (std::size_t idx = 0; idx < text.size(); idx += 2) {
        const char first = text[idx];
        const char second = idx + 1 < text.size() ? text[idx + 1] : '0';
        output.push_back(static_cast<std::uint8_t>((hex_value(second) << 4) | hex_value(first)));
    }
    return output;
}

std::string format_hex_bytes(const std::vector<std::uint8_t>& values, std::size_t start, std::size_t count)
{
    std::ostringstream stream;
    stream << std::uppercase << std::hex << std::setfill('0');
    for (std::size_t idx = 0; idx < count; ++idx) {
        stream << std::setw(2) << static_cast<int>(values[start + idx]);
    }
    return stream.str();
}

std::uint8_t swap_bits(std::uint8_t value)
{
    std::uint8_t result = 0;
    for (int idx = 0; idx < 8; ++idx) {
        result = static_cast<std::uint8_t>((result << 1U) | (value & 1U));
        value >>= 1U;
    }
    return result;
}

std::vector<std::uint8_t> dewhitening(const std::vector<std::uint8_t>& data, int channel)
{
    std::vector<std::uint8_t> output;
    output.reserve(data.size());
    int lfsr = static_cast<int>(swap_bits(static_cast<std::uint8_t>(channel))) | 2;

    for (auto item : data) {
        int d = static_cast<int>(swap_bits(item));
        for (int mask : {128, 64, 32, 16, 8, 4, 2, 1}) {
            if (lfsr & 0x80) {
                lfsr ^= 0x11;
                d ^= mask;
            }
            lfsr <<= 1;
        }
        output.push_back(swap_bits(static_cast<std::uint8_t>(d & 0xff)));
    }
    return output;
}

std::vector<std::uint8_t> crc24(const std::vector<std::uint8_t>& data, std::size_t length, int init)
{
    std::vector<std::uint8_t> ret = {
        static_cast<std::uint8_t>((init >> 16) & 0xff),
        static_cast<std::uint8_t>((init >> 8) & 0xff),
        static_cast<std::uint8_t>(init & 0xff),
    };

    const std::size_t count = std::min(length, data.size());
    for (std::size_t data_idx = 0; data_idx < count; ++data_idx) {
        int d = data[data_idx];
        for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
            const int t = (ret[0] >> 7) & 1;

            ret[0] = static_cast<std::uint8_t>((ret[0] << 1U) & 0xffU);
            if (ret[1] & 0x80U) {
                ret[0] |= 1U;
            }

            ret[1] = static_cast<std::uint8_t>((ret[1] << 1U) & 0xffU);
            if (ret[2] & 0x80U) {
                ret[1] |= 1U;
            }

            ret[2] = static_cast<std::uint8_t>((ret[2] << 1U) & 0xffU);

            if ((d & 1) != t) {
                ret[2] ^= 0x5bU;
                ret[1] ^= 0x06U;
            }

            d >>= 1;
        }
    }

    ret[0] = swap_bits(ret[0]);
    ret[1] = swap_bits(ret[1]);
    ret[2] = swap_bits(ret[2]);
    return ret;
}

std::string format_ble_address(const std::vector<std::uint8_t>& payload, std::size_t start)
{
    if (payload.size() < start + 6) {
        return "";
    }
    std::ostringstream stream;
    stream << std::uppercase << std::hex << std::setfill('0');
    for (std::size_t idx = 0; idx < 6; ++idx) {
        stream << std::setw(2) << static_cast<int>(payload[start + 5 - idx]);
    }
    return stream.str();
}

const char* adv_pdu_type_name(int pdu_type)
{
    switch (pdu_type) {
    case 0x0:
        return "ADV_IND";
    case 0x1:
        return "ADV_DIRECT_IND";
    case 0x2:
        return "ADV_NONCONN_IND";
    case 0x3:
        return "SCAN_REQ";
    case 0x4:
        return "SCAN_RSP";
    case 0x5:
        return "CONNECT_IND";
    case 0x6:
        return "ADV_SCAN_IND";
    case 0x7:
        return "ADV_EXT_IND";
    default:
        return "";
    }
}

void fill_adv_payload_info(
    BlePacketCandidate& candidate,
    const std::vector<std::uint8_t>& access_address,
    int channel,
    int header_flag,
    const std::vector<std::uint8_t>& payload)
{
    const std::string access_address_raw = format_hex_bytes(access_address, 0, access_address.size());
    if (channel != 37 && channel != 38 && channel != 39) {
        return;
    }
    if (hamming_distance_hex(access_address_raw, "D6BE898E") > 2) {
        return;
    }

    const int pdu_type = header_flag & 0x0f;
    const bool txaddr_random = ((header_flag >> 6) & 0x01) != 0;
    const bool rxaddr_random = ((header_flag >> 7) & 0x01) != 0;
    const std::string tx_type = txaddr_random ? "random" : "public";
    const std::string rx_type = rxaddr_random ? "random" : "public";

    const char* pdu_name = adv_pdu_type_name(pdu_type);
    candidate.ble_pdu_type = *pdu_name ? pdu_name : "ADV_TYPE_" + std::to_string(pdu_type);

    if ((pdu_type == 0x0 || pdu_type == 0x1 || pdu_type == 0x2 || pdu_type == 0x4 || pdu_type == 0x6)
        && payload.size() >= 6) {
        candidate.advertiser_address = format_ble_address(payload, 0);
        candidate.advertiser_address_type = tx_type;
        candidate.ble_device_address = candidate.advertiser_address;
    }

    if (pdu_type == 0x1 && payload.size() >= 12) {
        candidate.peer_address = format_ble_address(payload, 6);
        candidate.peer_address_type = rx_type;
    } else if (pdu_type == 0x3 && payload.size() >= 12) {
        candidate.peer_address = format_ble_address(payload, 0);
        candidate.peer_address_type = tx_type;
        candidate.advertiser_address = format_ble_address(payload, 6);
        candidate.advertiser_address_type = rx_type;
        candidate.ble_device_address = candidate.advertiser_address;
    } else if (pdu_type == 0x5 && payload.size() >= 12) {
        candidate.peer_address = format_ble_address(payload, 0);
        candidate.peer_address_type = tx_type;
        candidate.advertiser_address = format_ble_address(payload, 6);
        candidate.advertiser_address_type = rx_type;
        candidate.ble_device_address = candidate.advertiser_address;
    }

    if (candidate.ble_device_address.empty() && !candidate.advertiser_address.empty()) {
        candidate.ble_device_address = candidate.advertiser_address;
    }
}

bool valid_ble_data_header(int header_flag, int payload_len)
{
    const int llid = header_flag & 0x03;
    const int high_bits = (header_flag >> 5) & 0x07;
    if (llid == 0) {
        return false;
    }
    if (high_bits != 0b000 && high_bits != 0b001) {
        return false;
    }
    if (payload_len < 0 || payload_len > 251) {
        return false;
    }
    if ((llid == 0b10 || llid == 0b11) && payload_len == 0) {
        return false;
    }
    return true;
}

bool valid_ble_connection_access_address_impl(const std::string& access_address_raw)
{
    if (access_address_raw.size() != 8
        || !std::all_of(access_address_raw.begin(), access_address_raw.end(), [](unsigned char value) {
               return std::isxdigit(value) != 0;
           })) {
        return false;
    }
    if (hamming_distance_hex(access_address_raw, "D6BE898E") <= 1) {
        return false;
    }

    std::uint32_t value = 0;
    std::uint8_t first_octet = 0;
    bool all_octets_equal = true;
    for (std::size_t byte_idx = 0; byte_idx < 4; ++byte_idx) {
        const auto char_idx = byte_idx * 2;
        const auto octet = static_cast<std::uint8_t>(
            (hex_value(access_address_raw[char_idx]) << 4)
            | hex_value(access_address_raw[char_idx + 1]));
        if (byte_idx == 0) {
            first_octet = octet;
        } else if (octet != first_octet) {
            all_octets_equal = false;
        }
        value |= static_cast<std::uint32_t>(octet) << (byte_idx * 8U);
    }
    if (all_octets_equal) {
        return false;
    }

    int transitions = 0;
    int longest_run = 1;
    int current_run = 1;
    int previous = static_cast<int>(value & 1U);
    for (int bit_idx = 1; bit_idx < 32; ++bit_idx) {
        const int current = static_cast<int>((value >> bit_idx) & 1U);
        if (current != previous) {
            ++transitions;
            current_run = 1;
        } else {
            ++current_run;
            longest_run = std::max(longest_run, current_run);
        }
        previous = current;
    }
    if (transitions > 24 || longest_run > 6) {
        return false;
    }

    int most_significant_six_transitions = 0;
    previous = static_cast<int>((value >> 31U) & 1U);
    for (int bit_idx = 30; bit_idx >= 26; --bit_idx) {
        const int current = static_cast<int>((value >> bit_idx) & 1U);
        most_significant_six_transitions += current != previous ? 1 : 0;
        previous = current;
    }
    return most_significant_six_transitions >= 2;
}

}  // namespace

bool valid_ble_connection_access_address(const std::string& access_address_raw)
{
    return valid_ble_connection_access_address_impl(access_address_raw);
}

std::uint64_t build_bluetooth_sync_word(std::uint32_t lap)
{
    constexpr std::uint64_t bch_poly = 0260534236651ULL;
    constexpr std::uint64_t pn = 0x83848D96BBCC54FCULL;
    lap &= 0xFFFFFFU;
    const std::uint64_t barker = ((lap >> 23U) & 1U) ? 0x13U : 0x2cU;
    const std::uint64_t x = (barker << 24U) | lap;
    const std::uint64_t xtilde = (pn >> 34U) ^ x;
    const std::uint64_t parity = compute_remainder(xtilde << 34U, bch_poly);
    const std::uint64_t codeword = parity | (xtilde << 34U);
    return codeword ^ pn;
}

BrAccessCodeMatch find_br_access_code(const std::uint8_t* bits, std::int64_t bit_count)
{
    if (bit_count < 72) {
        return {-1, 0, false};
    }

    for (std::int64_t offset = 0; offset <= bit_count - 72; ++offset) {
        const bool preamble_a =
            bits[offset] == 1 && bits[offset + 1] == 0 && bits[offset + 2] == 1 && bits[offset + 3] == 0;
        const bool preamble_b =
            bits[offset] == 0 && bits[offset + 1] == 1 && bits[offset + 2] == 0 && bits[offset + 3] == 1;
        if (!preamble_a && !preamble_b) {
            continue;
        }

        const std::uint64_t barker = bits_to_uint_lsb(bits + offset + 62, 6);
        if (barker != 0x13U && barker != 0x2cU) {
            continue;
        }
        const std::uint32_t lap = static_cast<std::uint32_t>(bits_to_uint_lsb(bits + offset + 38, 24));
        const std::uint64_t code = bits_to_uint_lsb(bits + offset + 4, 34);
        const std::uint64_t captured = (barker << 58U) | (static_cast<std::uint64_t>(lap) << 34U) | code;
        if (build_bluetooth_sync_word(lap) == captured) {
            return {offset, lap, true};
        }
    }

    return {-1, 0, false};
}

std::vector<BrHeaderCandidate> decode_br_header_candidates(
    const std::uint8_t* raw_bits,
    std::int64_t bit_count,
    const int* candidate_uaps,
    std::int64_t candidate_uap_count,
    bool stop_after_first)
{
    constexpr std::int64_t access_code_len = 72;
    constexpr std::int64_t header_raw_len = 54;
    if (bit_count < access_code_len + header_raw_len) {
        return {};
    }

    const auto* raw_header_bits = raw_bits + access_code_len;
    const int header_fec = majority_vote_fec(raw_header_bits);
    std::vector<BrHeaderCandidate> candidates;

    for (std::int64_t uap_idx = 0; uap_idx < candidate_uap_count; ++uap_idx) {
        const int uap = candidate_uaps[uap_idx];
        for (int clk = 0; clk < 64; ++clk) {
            const int whitener = (clk & 0x3f) | 0x40;
            int header_dewhiten = header_fec;
            int temp_whitener = whitener;
            for (int bit_idx = 0; bit_idx < 18; ++bit_idx) {
                const int w_out = (temp_whitener >> 6) & 0x1;
                temp_whitener = ((temp_whitener << 1) & 0x7f) ^ (w_out | (w_out << 4));
                header_dewhiten ^= w_out << bit_idx;
            }
            if (check_hec(header_dewhiten, uap)) {
                int payload_prefix_count = 0;
                const std::uint16_t payload_prefix_bits = dewhiten_payload_prefix(
                    raw_bits + access_code_len + header_raw_len,
                    bit_count - access_code_len - header_raw_len,
                    temp_whitener,
                    &payload_prefix_count);
                candidates.push_back(
                    {
                        uap,
                        clk,
                        static_cast<std::uint32_t>(header_dewhiten),
                        temp_whitener,
                        bit_count - access_code_len - header_raw_len,
                        payload_prefix_bits,
                        payload_prefix_count,
                    });
                break;
            }
        }
        if (stop_after_first && !candidates.empty()) {
            break;
        }
    }

    return candidates;
}

std::vector<std::complex<float>> apply_fir_filter(
    const std::complex<float>* samples,
    std::int64_t sample_count,
    const float* taps,
    std::int64_t tap_count)
{
    std::vector<std::complex<float>> output(static_cast<std::size_t>(sample_count));
    for (std::int64_t idx = 0; idx < sample_count; ++idx) {
        float acc_re = 0.0F;
        float acc_im = 0.0F;
        const std::int64_t max_tap = std::min<std::int64_t>(tap_count, idx + 1);
        std::int64_t tap_idx = 0;
        for (; tap_idx + 3 < max_tap; tap_idx += 4) {
            const auto sample0 = samples[idx - tap_idx];
            const auto sample1 = samples[idx - tap_idx - 1];
            const auto sample2 = samples[idx - tap_idx - 2];
            const auto sample3 = samples[idx - tap_idx - 3];
            const float tap0 = taps[tap_idx];
            const float tap1 = taps[tap_idx + 1];
            const float tap2 = taps[tap_idx + 2];
            const float tap3 = taps[tap_idx + 3];
            acc_re += tap0 * sample0.real() + tap1 * sample1.real() +
                tap2 * sample2.real() + tap3 * sample3.real();
            acc_im += tap0 * sample0.imag() + tap1 * sample1.imag() +
                tap2 * sample2.imag() + tap3 * sample3.imag();
        }
        for (; tap_idx < max_tap; ++tap_idx) {
            const auto sample = samples[idx - tap_idx];
            const float tap = taps[tap_idx];
            acc_re += tap * sample.real();
            acc_im += tap * sample.imag();
        }
        output[static_cast<std::size_t>(idx)] = std::complex<float>(acc_re, acc_im);
    }
    return output;
}

double median_value(std::vector<double> values)
{
    if (values.empty()) {
        return NAN;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    double median = values[mid];
    if (values.size() % 2 == 0) {
        const auto lower = std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
        median = (*lower + median) / 2.0;
    }
    return median;
}

std::optional<double> estimate_br_cfo_hz(
    const std::complex<float>* samples,
    std::int64_t sample_count,
    const std::vector<std::uint8_t>& bits,
    std::int64_t bit_offset,
    int samples_per_bit,
    double sample_rate)
{
    if (sample_count < 2 || bits.empty()) {
        return std::nullopt;
    }

    std::vector<double> phase(static_cast<std::size_t>(sample_count));
    for (std::int64_t idx = 0; idx < sample_count; ++idx) {
        phase[static_cast<std::size_t>(idx)] = std::atan2(samples[idx].imag(), samples[idx].real());
        if (idx > 0) {
            double delta = phase[static_cast<std::size_t>(idx)] - phase[static_cast<std::size_t>(idx - 1)];
            if (delta > M_PI) {
                phase[static_cast<std::size_t>(idx)] -= 2.0 * M_PI;
            } else if (delta < -M_PI) {
                phase[static_cast<std::size_t>(idx)] += 2.0 * M_PI;
            }
        }
    }

    std::vector<double> inst_freq(static_cast<std::size_t>(sample_count - 1));
    for (std::int64_t idx = 1; idx < sample_count; ++idx) {
        inst_freq[static_cast<std::size_t>(idx - 1)] =
            (phase[static_cast<std::size_t>(idx)] - phase[static_cast<std::size_t>(idx - 1)]) *
            sample_rate / (2.0 * M_PI);
    }

    const std::int64_t bit_end = std::min<std::int64_t>(bit_offset + 72, static_cast<std::int64_t>(bits.size()));
    std::vector<double> freq_zero;
    std::vector<double> freq_one;
    for (std::int64_t bit_idx = bit_offset; bit_idx < bit_end; ++bit_idx) {
        const std::int64_t freq_idx = bit_idx * samples_per_bit + (samples_per_bit / 2);
        if (freq_idx >= static_cast<std::int64_t>(inst_freq.size())) {
            continue;
        }
        if (bits[static_cast<std::size_t>(bit_idx)] == 0) {
            freq_zero.push_back(inst_freq[static_cast<std::size_t>(freq_idx)]);
        } else {
            freq_one.push_back(inst_freq[static_cast<std::size_t>(freq_idx)]);
        }
    }
    if (freq_zero.size() < 3 || freq_one.size() < 3) {
        return std::nullopt;
    }
    return (median_value(std::move(freq_zero)) + median_value(std::move(freq_one))) / 2.0;
}

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
    int thread_count)
{
    if (segment_count < 0) {
        throw std::invalid_argument("segment_count must be non-negative");
    }
    if (tap_count <= 0) {
        throw std::invalid_argument("tap_count must be positive");
    }
    if (samples_per_bit <= 0) {
        throw std::invalid_argument("samples_per_bit must be positive");
    }
    if (thread_count < 1) {
        throw std::invalid_argument("thread_count must be positive");
    }
    for (std::int64_t segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
        const std::int64_t offset = offsets[segment_idx];
        const std::int64_t length = lengths[segment_idx];
        if (offset < 0 || length < 0 || offset + length > sample_count) {
            throw std::out_of_range("segment offset/length is outside iq_concat");
        }
    }

    std::vector<float> lpf_taps_f32(static_cast<std::size_t>(tap_count));
    for (std::int64_t tap_idx = 0; tap_idx < tap_count; ++tap_idx) {
        lpf_taps_f32[static_cast<std::size_t>(tap_idx)] = static_cast<float>(lpf_taps[tap_idx]);
    }
    std::vector<int> uaps(256);
    std::iota(uaps.begin(), uaps.end(), 0);

    std::vector<std::optional<BrPacketCandidate>> candidate_slots(static_cast<std::size_t>(segment_count));
    auto process_range = [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t segment_idx = begin; segment_idx < end; ++segment_idx) {
            const std::int64_t offset = offsets[segment_idx];
            const std::int64_t length = lengths[segment_idx];
            const auto* segment = samples + offset;
            auto filtered = apply_fir_filter(segment, length, lpf_taps_f32.data(), tap_count);
            auto demod = gfsk_demodulate(filtered.data(), static_cast<std::int64_t>(filtered.size()), gain);
            auto bits_1m = decision_bits(
                demod.data(),
                static_cast<std::int64_t>(demod.size()),
                samples_per_bit);
            const auto access = find_br_access_code(bits_1m.data(), static_cast<std::int64_t>(bits_1m.size()));
            if (!access.valid || access.offset < 0) {
                continue;
            }

            std::vector<std::uint8_t> raw_bits(
                bits_1m.begin() + static_cast<std::ptrdiff_t>(access.offset),
                bits_1m.end());
            auto header_candidates = decode_br_header_candidates(
                raw_bits.data(),
                static_cast<std::int64_t>(raw_bits.size()),
                uaps.data(),
                static_cast<std::int64_t>(uaps.size()),
                false);
            std::vector<std::uint8_t> payload_bits;
            if (raw_bits.size() > 72 + 54) {
                payload_bits.assign(raw_bits.begin() + 72 + 54, raw_bits.end());
            }

            BrPacketCandidate candidate;
            candidate.sample_index = sample_indices[segment_idx] + access.offset * samples_per_bit;
            candidate.segment_index = segment_indices[segment_idx];
            candidate.lap = access.lap;
            candidate.rssi = compute_rssi_db(filtered.data(), static_cast<std::int64_t>(filtered.size()), -60.0);
            candidate.cfo_hz = estimate_br_cfo_hz(
                filtered.data(),
                static_cast<std::int64_t>(filtered.size()),
                bits_1m,
                access.offset,
                samples_per_bit,
                sample_rate);
            candidate.raw_bits_len = static_cast<std::int64_t>(raw_bits.size());
            candidate.header_candidates = std::move(header_candidates);
            candidate.payload_bits = std::move(payload_bits);
            candidate_slots[static_cast<std::size_t>(segment_idx)] = std::move(candidate);
        }
    };

    const std::int64_t effective_threads = std::min<std::int64_t>(thread_count, segment_count);
    if (effective_threads <= 1) {
        process_range(0, segment_count);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(effective_threads));
        const std::int64_t chunk = (segment_count + effective_threads - 1) / effective_threads;
        for (std::int64_t thread_idx = 0; thread_idx < effective_threads; ++thread_idx) {
            const std::int64_t begin = thread_idx * chunk;
            const std::int64_t end = std::min<std::int64_t>(segment_count, begin + chunk);
            if (begin >= end) {
                break;
            }
            workers.emplace_back(process_range, begin, end);
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }

    std::vector<BrPacketCandidate> candidates;
    candidates.reserve(static_cast<std::size_t>(segment_count));
    for (auto& candidate : candidate_slots) {
        if (candidate) {
            candidates.push_back(std::move(*candidate));
        }
    }
    return candidates;
}

BrPacketDecision process_br_header_candidates(
    const std::vector<BrHeaderCandidate>& candidates,
    std::int64_t raw_bits_len,
    std::optional<int> locked_uap,
    int miss_count,
    int max_miss,
    bool retry_after_unlock)
{
    constexpr std::int64_t access_code_len = 72;
    constexpr std::int64_t header_raw_len = 54;
    if (raw_bits_len < access_code_len + header_raw_len) {
        return {"Too_Short", -1, -1, "", -1, 0, -1, locked_uap, miss_count, false};
    }

    const BrHeaderCandidate* best = nullptr;
    if (locked_uap.has_value()) {
        for (const auto& candidate : candidates) {
            if (candidate.uap == *locked_uap) {
                best = &candidate;
                break;
            }
        }
    } else if (!candidates.empty()) {
        best = &candidates.front();
    }

    std::string status;
    if (best == nullptr && locked_uap.has_value()) {
        ++miss_count;
        if (miss_count >= max_miss) {
            locked_uap.reset();
            miss_count = 0;
            if (retry_after_unlock && !candidates.empty()) {
                best = &candidates.front();
            }
        }
    }

    if (best == nullptr) {
        return {"Searching", -1, -1, "", -1, 0, -1, locked_uap, miss_count, false};
    }

    status = locked_uap.has_value() ? "Locked" : "New_Lock";
    locked_uap = best->uap;
    miss_count = 0;

    const int type_val = static_cast<int>((best->header >> 3U) & 0x0fU);
    const auto [type_name, h_mode] = get_packet_type_info(type_val);
    int length = 0;
    if (h_mode == 1 && best->payload_prefix_count >= 8) {
        length = (best->payload_prefix_bits >> 3U) & 0x1fU;
    } else if (h_mode == 2 && best->payload_prefix_count >= 16) {
        length = (best->payload_prefix_bits >> 3U) & 0x1ffU;
    } else if (h_mode == -1) {
        if (type_val == 5) {
            length = 10;
        } else if (type_val == 6) {
            length = 20;
        } else if (type_val == 7) {
            length = 30;
        }
    }

    return {
        status,
        best->uap,
        best->clk,
        type_name,
        type_val,
        length,
        compute_br_air_total_bytes(type_val, length),
        locked_uap,
        miss_count,
        true,
    };
}

std::vector<BrPacketDecision> process_br_header_candidate_sequence(
    const std::vector<std::vector<BrHeaderCandidate>>& candidate_batches,
    const std::vector<std::int64_t>& raw_bits_lens,
    std::optional<int> locked_uap,
    int miss_count,
    int max_miss,
    bool retry_after_unlock)
{
    if (candidate_batches.size() != raw_bits_lens.size()) {
        throw std::invalid_argument("candidate_batches and raw_bits_lens must have the same size");
    }

    std::vector<BrPacketDecision> decisions;
    decisions.reserve(candidate_batches.size());
    for (std::size_t idx = 0; idx < candidate_batches.size(); ++idx) {
        auto decision = process_br_header_candidates(
            candidate_batches[idx],
            raw_bits_lens[idx],
            locked_uap,
            miss_count,
            max_miss,
            retry_after_unlock);
        locked_uap = decision.locked_uap;
        miss_count = decision.miss_count;
        decisions.push_back(std::move(decision));
    }
    return decisions;
}

std::optional<std::string> detect_ble_access_address(
    const std::uint8_t* bits,
    std::int64_t bit_count,
    const std::string& ble_mode)
{
    for (std::int64_t bit_offset = 0; bit_offset < 4; ++bit_offset) {
        if (bit_offset >= bit_count) {
            break;
        }
        const auto buffer = bits_to_hex_buffer(bits + bit_offset, bit_count - bit_offset);
        const auto [pre_pos, detected_mode] = find_patterns(buffer, ble_mode);
        if (pre_pos == -1) {
            continue;
        }

        std::size_t preamble_len = 0;
        if (detected_mode == "2M") {
            preamble_len = 2;
        } else if (detected_mode == "1M") {
            preamble_len = 1;
        } else {
            continue;
        }

        const auto bytes = reverse_bytes_to_ints(buffer.substr(static_cast<std::size_t>(pre_pos)));
        if (bytes.size() < preamble_len + 4) {
            continue;
        }
        return format_hex_bytes(bytes, preamble_len, 4);
    }
    return std::nullopt;
}

static std::vector<BlePacketCandidate> parse_ble_packet_bits_candidates(
    const std::uint8_t* bits,
    std::int64_t bit_count,
    const std::string& ble_mode,
    std::int64_t iq_len,
    int channel)
{
    const double ble_mod_rate = ble_mode == "2M" ? 2e6 : 1e6;
    const int samples_per_bit = static_cast<int>(4e6 / ble_mod_rate);
    const int tail_sample = samples_per_bit * 4;
    const int len_exclude_payload = 1 + 4 + 2 + 3;

    std::vector<BlePacketCandidate> candidates;
    for (std::int64_t bit_offset = 0; bit_offset < 4; ++bit_offset) {
        if (bit_offset >= bit_count) {
            break;
        }

        const auto buffer = bits_to_hex_buffer(bits + bit_offset, bit_count - bit_offset);
        const auto preamble_positions = find_all_pattern_positions(buffer, ble_mode);
        for (const auto pre_pos : preamble_positions) {
            const std::size_t preamble_len = ble_mode == "2M" ? 2U : 1U;
            const auto ble_pkt_buffer = reverse_bytes_to_ints(
                buffer.substr(static_cast<std::size_t>(pre_pos)));
            if (ble_pkt_buffer.size() < 10 + preamble_len) {
                continue;
            }

            std::vector<std::uint8_t> access_address(
            ble_pkt_buffer.begin() + static_cast<std::ptrdiff_t>(preamble_len),
            ble_pkt_buffer.begin() + static_cast<std::ptrdiff_t>(preamble_len + 4));
            const auto access_address_raw = format_hex_bytes(access_address, 0, access_address.size());
            if ((channel == 37 || channel == 38 || channel == 39)
                && hamming_distance_hex(access_address_raw, "D6BE898E") > 2) {
                continue;
            }
            if (channel != 37 && channel != 38 && channel != 39
                && !valid_ble_connection_access_address(access_address_raw)) {
                continue;
            }
            const auto header_payload_offset = preamble_len + 4;
            std::vector<std::uint8_t> whitened_header_payload_crc(
                ble_pkt_buffer.size() - header_payload_offset);
            std::copy_n(
                ble_pkt_buffer.data() + header_payload_offset,
                whitened_header_payload_crc.size(),
                whitened_header_payload_crc.data());
            const auto header_payload_crc = dewhitening(
                whitened_header_payload_crc, channel);
            if (header_payload_crc.size() < 2) {
                continue;
            }

            const int header_flag = header_payload_crc[0];
            const int payload_len = header_payload_crc[1];
            const int bit_flag = (header_flag >> 5) & 0x07;
            if (channel != 37 && channel != 38 && channel != 39
                && !valid_ble_data_header(header_flag, payload_len)) {
                continue;
            }
            const auto available_payload = header_payload_crc.size() > 2 ? header_payload_crc.size() - 2 : 0;
            const auto actual_payload_len = std::min<std::size_t>(static_cast<std::size_t>(payload_len), available_payload);
            std::vector<std::uint8_t> payload(
            header_payload_crc.begin() + 2,
            header_payload_crc.begin() + 2 + static_cast<std::ptrdiff_t>(actual_payload_len));
            const auto whitened_pdu_start = preamble_len + 4;
            const auto requested_pdu_len = static_cast<std::size_t>(2 + payload_len);
            const auto available_pdu_len =
            ble_pkt_buffer.size() > whitened_pdu_start
                ? ble_pkt_buffer.size() - whitened_pdu_start
                : 0;
            const auto whitened_pdu_len = std::min(requested_pdu_len, available_pdu_len);

            const auto preamble_sample_offset =
                (bit_offset + pre_pos * 4) * samples_per_bit;
            const auto available_samples = std::max<std::int64_t>(
                0, iq_len - preamble_sample_offset);
            const auto available_packet_bytes = static_cast<std::size_t>(
                available_samples / (8 * samples_per_bit));
            const auto captured_header_payload_crc_len =
                available_packet_bytes > preamble_len + 4
                    ? std::min<std::size_t>(
                        header_payload_crc.size(),
                        available_packet_bytes - preamble_len - 4)
                    : 0U;
            const auto required_samples =
                (len_exclude_payload + payload_len) * 8 * samples_per_bit + tail_sample;
            double score = static_cast<double>(
                std::max<std::int64_t>(0, required_samples - available_samples))
                / static_cast<double>(8 * samples_per_bit);

            if (channel != 37 && channel != 38 && channel != 39) {
                if (bit_flag != 0b000 && bit_flag != 0b001) {
                    score += 5.0;
                }
            }

            const auto _crc_cal = crc24(header_payload_crc, static_cast<std::size_t>(payload_len + 2), 0x97a38e);
            (void)_crc_cal;

            BlePacketCandidate candidate;
            candidate.access_address = access_address_raw;
            candidate.payload_len = payload_len;
            candidate.score = score;
            candidate.sample_offset = preamble_sample_offset;
            candidate.whitened_pdu_hex = format_hex_bytes(
            ble_pkt_buffer,
            whitened_pdu_start,
            whitened_pdu_len);
            const auto pdu_end = static_cast<std::size_t>(2 + payload_len);
            const auto dewhitened_pdu_len = std::min(pdu_end, header_payload_crc.size());
            candidate.dewhitened_pdu_hex = format_hex_bytes(
                header_payload_crc, 0, dewhitened_pdu_len);
            candidate.captured_crc_hex.clear();
            candidate.post_crc_hex.clear();
            candidate.crc_and_post_crc_hex.clear();
            candidate.crc_capture_status = "truncated";
            if (payload_len < 0 || payload_len > 255) {
                candidate.dewhitened_pdu_hex.clear();
                candidate.crc_capture_status = "invalid_length";
            } else if (captured_header_payload_crc_len >= pdu_end + 3) {
                candidate.captured_crc_hex = format_hex_bytes(
                    header_payload_crc, pdu_end, 3);
                candidate.crc_and_post_crc_hex = format_hex_bytes(
                    header_payload_crc, pdu_end, captured_header_payload_crc_len - pdu_end);
                if (captured_header_payload_crc_len > pdu_end + 3) {
                    candidate.post_crc_hex = format_hex_bytes(
                        header_payload_crc,
                        pdu_end + 3,
                        captured_header_payload_crc_len - pdu_end - 3);
                }
                candidate.crc_capture_status = "ok";
            }
            fill_adv_payload_info(candidate, access_address, channel, header_flag, payload);

            candidates.push_back(std::move(candidate));
        }
    }

    return candidates;
}

std::optional<BlePacketCandidate> parse_ble_packet_bits(
    const std::uint8_t* bits,
    std::int64_t bit_count,
    const std::string& ble_mode,
    std::int64_t iq_len,
    int channel)
{
    auto candidates = parse_ble_packet_bits_candidates(
        bits, bit_count, ble_mode, iq_len, channel);
    std::optional<BlePacketCandidate> best;
    for (auto& candidate : candidates) {
        if (!best || candidate.score < best->score) {
            best = std::move(candidate);
        }
    }
    return best;
}

static double estimate_gfsk_discriminator_bias(const std::vector<double>& demod)
{
    std::vector<double> values;
    values.reserve(demod.size());
    std::copy_if(demod.begin(), demod.end(), std::back_inserter(values), [](double value) {
        return std::isfinite(value);
    });
    if (values.size() < 16) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto trim = values.size() / 50;
    if (trim > 0 && trim * 2 < values.size() - 8) {
        values = std::vector<double>(values.begin() + trim, values.end() - trim);
    }
    double centers[2] = {
        values[values.size() / 4],
        values[(values.size() * 3) / 4],
    };
    for (int iteration = 0; iteration < 12; ++iteration) {
        std::vector<double> clusters[2];
        for (const auto value : values) {
            const int cluster = std::abs(value - centers[0]) <= std::abs(value - centers[1]) ? 0 : 1;
            clusters[cluster].push_back(value);
        }
        if (clusters[0].size() < 4 || clusters[1].size() < 4) {
            return 0.0;
        }
        double next[2];
        for (int cluster = 0; cluster < 2; ++cluster) {
            auto& items = clusters[cluster];
            const auto middle = items.begin() + items.size() / 2;
            std::nth_element(items.begin(), middle, items.end());
            next[cluster] = *middle;
        }
        if (std::abs(next[0] - centers[0]) < 1e-9
            && std::abs(next[1] - centers[1]) < 1e-9) {
            centers[0] = next[0];
            centers[1] = next[1];
            break;
        }
        centers[0] = next[0];
        centers[1] = next[1];
    }
    return (centers[0] + centers[1]) / 2.0;
}

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
    int thread_count)
{
    if (segment_count < 0) {
        throw std::invalid_argument("segment_count must be non-negative");
    }
    if (thread_count < 1) {
        throw std::invalid_argument("thread_count must be positive");
    }
    for (std::int64_t segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
        const std::int64_t offset = offsets[segment_idx];
        const std::int64_t length = lengths[segment_idx];
        const std::int64_t score_length = score_lengths[segment_idx];
        if (offset < 0 || length < 0 || score_length < 0 || score_length > length ||
            offset + length > sample_count) {
            throw std::out_of_range("segment offset/length is outside iq_concat");
        }
    }

    std::vector<std::vector<BlePacketCandidate>> packet_slots(static_cast<std::size_t>(segment_count));
    auto process_range = [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t segment_idx = begin; segment_idx < end; ++segment_idx) {
            const std::int64_t offset = offsets[segment_idx];
            const std::int64_t length = lengths[segment_idx];
            const std::int64_t score_length = score_lengths[segment_idx];

            const auto* segment = samples + offset;
            auto demod = gfsk_demodulate(segment, length, gain);
            const double discriminator_bias = estimate_gfsk_discriminator_bias(demod);
            std::vector<BlePacketCandidate> segment_candidates;
            for (const auto& mode : {std::pair{"2M", 2}, std::pair{"1M", 4}}) {
                for (int phase = 0; phase < mode.second; ++phase) {
                    std::vector<std::uint8_t> point_bits;
                    std::vector<std::uint8_t> integrated_bits;
                    std::vector<std::uint8_t> centered_integrated_bits;
                    for (std::int64_t idx = phase; idx < static_cast<std::int64_t>(demod.size()); idx += mode.second) {
                        point_bits.push_back(demod[static_cast<std::size_t>(idx)] > 0.0 ? 1 : 0);
                        const auto stop = std::min<std::int64_t>(
                            static_cast<std::int64_t>(demod.size()), idx + mode.second);
                        const double symbol_sum = std::accumulate(
                            demod.begin() + idx, demod.begin() + stop, 0.0);
                        integrated_bits.push_back(symbol_sum > 0.0 ? 1 : 0);
                        centered_integrated_bits.push_back(
                            symbol_sum - discriminator_bias * (stop - idx) > 0.0 ? 1 : 0);
                    }
                    for (const auto* bits : {&point_bits, &integrated_bits, &centered_integrated_bits}) {
                        auto phase_candidates = parse_ble_packet_bits_candidates(
                            bits->data(),
                            static_cast<std::int64_t>(bits->size()),
                            mode.first,
                            std::max<std::int64_t>(0, score_length - phase),
                            channel);
                        for (auto& packet : phase_candidates) {
                            packet.sample_offset += phase;
                            if (packet.score < score_threshold) {
                                segment_candidates.push_back(std::move(packet));
                            }
                        }
                    }
                }
            }

            std::sort(
                segment_candidates.begin(),
                segment_candidates.end(),
                [](const auto& lhs, const auto& rhs) {
                    if (lhs.sample_offset != rhs.sample_offset) {
                        return lhs.sample_offset < rhs.sample_offset;
                    }
                    if (lhs.access_address != rhs.access_address) {
                        return lhs.access_address < rhs.access_address;
                    }
                    return lhs.score < rhs.score;
                });
            std::vector<BlePacketCandidate> deduplicated;
            const double segment_rssi = compute_rssi_db(segment, length, -30.0);
            for (auto& packet : segment_candidates) {
                packet.sample_index = sample_indices[segment_idx] + packet.sample_offset;
                packet.segment_index = segment_idx;
                packet.rssi = segment_rssi;
                auto duplicate = std::find_if(
                    deduplicated.rbegin(),
                    deduplicated.rend(),
                    [&](const auto& existing) {
                        return packet.sample_offset - existing.sample_offset <= 4
                            && packet.access_address == existing.access_address;
                    });
                if (duplicate != deduplicated.rend()) {
                    if (packet.score < duplicate->score) {
                        *duplicate = std::move(packet);
                    }
                    continue;
                }
                deduplicated.push_back(std::move(packet));
            }
            packet_slots[static_cast<std::size_t>(segment_idx)] = std::move(deduplicated);
        }
    };

    const std::int64_t effective_threads = std::min<std::int64_t>(thread_count, segment_count);
    if (effective_threads <= 1) {
        process_range(0, segment_count);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(effective_threads));
        const std::int64_t chunk = (segment_count + effective_threads - 1) / effective_threads;
        for (std::int64_t thread_idx = 0; thread_idx < effective_threads; ++thread_idx) {
            const std::int64_t begin = thread_idx * chunk;
            const std::int64_t end = std::min<std::int64_t>(segment_count, begin + chunk);
            if (begin >= end) {
                break;
            }
            workers.emplace_back(process_range, begin, end);
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }

    std::vector<BlePacketCandidate> packets;
    for (auto& slot : packet_slots) {
        for (auto& packet : slot) {
            packets.push_back(std::move(packet));
        }
    }
    return packets;
}

double compute_rssi_db(
    const std::complex<float>* samples,
    std::int64_t sample_count,
    double gain_offset_db)
{
    if (sample_count <= 0) {
        return -INFINITY;
    }

    double power_sum = 0.0;
    for (std::int64_t idx = 0; idx < sample_count; ++idx) {
        const double re = static_cast<double>(samples[idx].real());
        const double im = static_cast<double>(samples[idx].imag());
        power_sum += re * re + im * im;
    }
    return 10.0 * std::log10(power_sum / static_cast<double>(sample_count)) + gain_offset_db;
}

std::vector<SegmentSummary> summarize_segments(
    const std::complex<float>* samples,
    std::int64_t sample_count,
    const std::int64_t* offsets,
    const std::int64_t* lengths,
    std::int64_t segment_count)
{
    if (segment_count < 0) {
        throw std::invalid_argument("segment_count must be non-negative");
    }

    std::vector<SegmentSummary> summaries;
    summaries.reserve(static_cast<std::size_t>(segment_count));

    for (std::int64_t segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
        const std::int64_t offset = offsets[segment_idx];
        const std::int64_t length = lengths[segment_idx];
        if (offset < 0 || length < 0 || offset + length > sample_count) {
            throw std::out_of_range("segment offset/length is outside iq_concat");
        }

        double power_sum = 0.0;
        for (std::int64_t idx = 0; idx < length; ++idx) {
            const auto sample = samples[offset + idx];
            const double re = static_cast<double>(sample.real());
            const double im = static_cast<double>(sample.imag());
            power_sum += re * re + im * im;
        }

        summaries.push_back({offset, length, power_sum});
    }

    return summaries;
}

}  // namespace bt_native
