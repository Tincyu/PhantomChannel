#include <cstddef>
#include <complex>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bt_native/signal.hpp"
#include "bt_native/version.hpp"

namespace py = pybind11;

namespace {

std::optional<int> optional_int_from_python(py::object value);

double sum_float32(py::array_t<float, py::array::c_style | py::array::forcecast> values)
{
    const py::buffer_info info = values.request();
    const auto* data = static_cast<const float*>(info.ptr);

    double total = 0.0;
    for (std::size_t idx = 0; idx < static_cast<std::size_t>(info.size); ++idx) {
        total += static_cast<double>(data[idx]);
    }
    return total;
}

py::dict ble_candidate_to_dict(const bt_native::BlePacketCandidate& candidate)
{
    py::dict output;
    py::list access_address;
    for (std::size_t idx = 0; idx < candidate.access_address.size(); idx += 2) {
        access_address.append("0x" + candidate.access_address.substr(idx, 2));
    }
    output["access_address"] = access_address;
    output["pkt_len"] = candidate.payload_len;
    output["score"] = candidate.score;
    output["ble_pdu_type"] = candidate.ble_pdu_type;
    output["whitened_pdu_hex"] = candidate.whitened_pdu_hex;
    output["dewhitened_pdu_hex"] = candidate.dewhitened_pdu_hex;
    output["captured_crc_hex"] = candidate.captured_crc_hex;
    output["post_crc_hex"] = candidate.post_crc_hex;
    output["crc_and_post_crc_hex"] = candidate.crc_and_post_crc_hex;
    output["crc_capture_status"] = candidate.crc_capture_status;
    output["advertiser_address"] = candidate.advertiser_address;
    output["advertiser_address_type"] = candidate.advertiser_address_type;
    output["peer_address"] = candidate.peer_address;
    output["peer_address_type"] = candidate.peer_address_type;
    output["ble_device_address"] = candidate.ble_device_address;
    output["sample_offset"] = candidate.sample_offset;
    output["sample_index"] = candidate.sample_index;
    output["segment_index"] = candidate.segment_index;
    output["rssi"] = candidate.rssi;
    return output;
}

struct PackedSegmentInput {
    std::vector<py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>> buffers;
    std::vector<std::complex<float>> samples;
    std::vector<std::int64_t> offsets;
    std::vector<std::int64_t> lengths;
    std::vector<std::int64_t> score_lengths;
    std::vector<std::int64_t> sample_indices;
    std::vector<std::int64_t> segment_indices;
};

PackedSegmentInput pack_segment_buffers_complex64(
    py::list buffers,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> buffer_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> buffer_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> score_lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sample_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> segment_indices)
{
    const py::buffer_info buffer_index_info = buffer_indices.request();
    const py::buffer_info offset_info = buffer_offsets.request();
    const py::buffer_info length_info = lengths.request();
    const py::buffer_info score_length_info = score_lengths.request();
    const py::buffer_info sample_index_info = sample_indices.request();
    const py::buffer_info segment_index_info = segment_indices.request();
    if (buffer_index_info.size != offset_info.size ||
        buffer_index_info.size != length_info.size ||
        buffer_index_info.size != score_length_info.size ||
        buffer_index_info.size != sample_index_info.size ||
        buffer_index_info.size != segment_index_info.size) {
        throw std::invalid_argument(
            "buffer_indices, offsets, lengths, score_lengths, sample_indices, and segment_indices must have the same size");
    }

    PackedSegmentInput packed;
    const auto segment_count = static_cast<std::size_t>(buffer_index_info.size);
    const auto buffer_count = static_cast<std::size_t>(py::len(buffers));
    packed.buffers.reserve(buffer_count);
    std::vector<py::buffer_info> buffer_infos;
    buffer_infos.reserve(buffer_count);
    for (const py::handle item : buffers) {
        packed.buffers.emplace_back(
            py::reinterpret_borrow<py::object>(item));
        buffer_infos.push_back(packed.buffers.back().request());
        if (buffer_infos.back().ndim != 1) {
            throw std::invalid_argument("each segment buffer must be a 1-D complex64 array");
        }
    }

    const auto* buffer_index_data = static_cast<const std::int64_t*>(buffer_index_info.ptr);
    const auto* offset_data = static_cast<const std::int64_t*>(offset_info.ptr);
    const auto* length_data = static_cast<const std::int64_t*>(length_info.ptr);
    const auto* score_length_data = static_cast<const std::int64_t*>(score_length_info.ptr);
    const auto* sample_index_data = static_cast<const std::int64_t*>(sample_index_info.ptr);
    const auto* segment_index_data = static_cast<const std::int64_t*>(segment_index_info.ptr);

    std::int64_t total_samples = 0;
    packed.offsets.reserve(segment_count);
    packed.lengths.reserve(segment_count);
    packed.score_lengths.reserve(segment_count);
    packed.sample_indices.reserve(segment_count);
    packed.segment_indices.reserve(segment_count);
    for (std::size_t idx = 0; idx < segment_count; ++idx) {
        const auto buffer_index = buffer_index_data[idx];
        const auto offset = offset_data[idx];
        const auto length = length_data[idx];
        if (buffer_index < 0 || static_cast<std::size_t>(buffer_index) >= buffer_infos.size()) {
            throw std::out_of_range("segment descriptor buffer_index is out of range");
        }
        if (offset < 0 || length < 0 ||
            offset + length > static_cast<std::int64_t>(buffer_infos[static_cast<std::size_t>(buffer_index)].size)) {
            throw std::out_of_range("segment descriptor offset/length exceeds buffer bounds");
        }
        packed.offsets.push_back(total_samples);
        packed.lengths.push_back(length);
        packed.score_lengths.push_back(score_length_data[idx]);
        packed.sample_indices.push_back(sample_index_data[idx]);
        packed.segment_indices.push_back(segment_index_data[idx]);
        total_samples += length;
    }

    packed.samples.resize(static_cast<std::size_t>(total_samples));
    for (std::size_t idx = 0; idx < segment_count; ++idx) {
        const auto buffer_index = static_cast<std::size_t>(buffer_index_data[idx]);
        const auto offset = static_cast<std::size_t>(offset_data[idx]);
        const auto length = static_cast<std::size_t>(length_data[idx]);
        const auto output_offset = static_cast<std::size_t>(packed.offsets[idx]);
        const auto* source = static_cast<const std::complex<float>*>(buffer_infos[buffer_index].ptr);
        std::copy(
            source + offset,
            source + offset + length,
            packed.samples.data() + output_offset);
    }
    return packed;
}

py::array_t<double> gfsk_demodulate_complex64(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> samples,
    double gain)
{
    const py::buffer_info info = samples.request();
    const auto* data = static_cast<const std::complex<float>*>(info.ptr);
    auto demod = bt_native::gfsk_demodulate(data, static_cast<std::int64_t>(info.size), gain);

    py::array_t<double> output(demod.size());
    auto output_info = output.request();
    auto* output_data = static_cast<double*>(output_info.ptr);
    std::copy(demod.begin(), demod.end(), output_data);
    return output;
}

py::array_t<std::uint8_t> decision_bits_float64(
    py::array_t<double, py::array::c_style | py::array::forcecast> freq_dev,
    std::int64_t samples_per_symbol)
{
    const py::buffer_info info = freq_dev.request();
    const auto* data = static_cast<const double*>(info.ptr);
    auto bits = bt_native::decision_bits(data, static_cast<std::int64_t>(info.size), samples_per_symbol);

    py::array_t<std::uint8_t> output(bits.size());
    auto output_info = output.request();
    auto* output_data = static_cast<std::uint8_t*>(output_info.ptr);
    std::copy(bits.begin(), bits.end(), output_data);
    return output;
}

py::object detect_ble_access_address_bits(
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> bits,
    const std::string& ble_mode)
{
    const py::buffer_info info = bits.request();
    const auto* data = static_cast<const std::uint8_t*>(info.ptr);
    const auto result = bt_native::detect_ble_access_address(
        data,
        static_cast<std::int64_t>(info.size),
        ble_mode);
    if (!result) {
        return py::none();
    }
    return py::str(*result);
}

py::object parse_ble_packet_bits(
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> bits,
    const std::string& ble_mode,
    std::int64_t iq_len,
    int channel)
{
    const py::buffer_info info = bits.request();
    const auto* data = static_cast<const std::uint8_t*>(info.ptr);
    const auto result = bt_native::parse_ble_packet_bits(
        data,
        static_cast<std::int64_t>(info.size),
        ble_mode,
        iq_len,
        channel);
    if (!result) {
        return py::none();
    }

    return ble_candidate_to_dict(*result);
}

py::list parse_ble_segments_complex64(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> samples,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> score_lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sample_indices,
    double gain,
    double score_threshold,
    int channel,
    int thread_count)
{
    const py::buffer_info sample_info = samples.request();
    const py::buffer_info offset_info = offsets.request();
    const py::buffer_info length_info = lengths.request();
    const py::buffer_info score_length_info = score_lengths.request();
    const py::buffer_info sample_index_info = sample_indices.request();
    if (offset_info.size != length_info.size || length_info.size != score_length_info.size ||
        offset_info.size != sample_index_info.size) {
        throw std::invalid_argument("offsets, lengths, score_lengths, and sample_indices must have the same size");
    }

    const auto* sample_data = static_cast<const std::complex<float>*>(sample_info.ptr);
    const auto* offset_data = static_cast<const std::int64_t*>(offset_info.ptr);
    const auto* length_data = static_cast<const std::int64_t*>(length_info.ptr);
    const auto* score_length_data = static_cast<const std::int64_t*>(score_length_info.ptr);
    const auto* sample_index_data = static_cast<const std::int64_t*>(sample_index_info.ptr);
    const auto packets = bt_native::parse_ble_segments(
        sample_data,
        static_cast<std::int64_t>(sample_info.size),
        offset_data,
        length_data,
        score_length_data,
        sample_index_data,
        static_cast<std::int64_t>(offset_info.size),
        gain,
        score_threshold,
        channel,
        thread_count);

    py::list output;
    for (const auto& packet : packets) {
        output.append(ble_candidate_to_dict(packet));
    }
    return output;
}

py::list parse_ble_segment_buffers_complex64(
    py::list buffers,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> buffer_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> buffer_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> score_lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sample_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> segment_indices,
    double gain,
    double score_threshold,
    int channel,
    int thread_count)
{
    const auto packed = pack_segment_buffers_complex64(
        buffers,
        buffer_indices,
        buffer_offsets,
        lengths,
        score_lengths,
        sample_indices,
        segment_indices);
    const auto packets = bt_native::parse_ble_segments(
        packed.samples.data(),
        static_cast<std::int64_t>(packed.samples.size()),
        packed.offsets.data(),
        packed.lengths.data(),
        packed.score_lengths.data(),
        packed.sample_indices.data(),
        static_cast<std::int64_t>(packed.offsets.size()),
        gain,
        score_threshold,
        channel,
        thread_count);

    py::list output;
    for (const auto& packet : packets) {
        output.append(ble_candidate_to_dict(packet));
    }
    return output;
}

std::uint64_t build_bluetooth_sync_word(std::uint32_t lap)
{
    return bt_native::build_bluetooth_sync_word(lap);
}

py::dict find_br_access_code_bits(
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> bits)
{
    const py::buffer_info info = bits.request();
    const auto* data = static_cast<const std::uint8_t*>(info.ptr);
    const auto result = bt_native::find_br_access_code(
        data,
        static_cast<std::int64_t>(info.size));
    py::dict output;
    output["offset"] = result.offset;
    output["lap"] = result.valid ? py::cast(result.lap) : py::none();
    output["valid"] = result.valid;
    return output;
}

py::list build_br_packet_candidates_complex64(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> samples,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sample_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> segment_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> lpf_taps,
    double gain,
    int samples_per_bit,
    double sample_rate,
    int thread_count)
{
    const py::buffer_info sample_info = samples.request();
    const py::buffer_info offset_info = offsets.request();
    const py::buffer_info length_info = lengths.request();
    const py::buffer_info sample_index_info = sample_indices.request();
    const py::buffer_info segment_index_info = segment_indices.request();
    const py::buffer_info tap_info = lpf_taps.request();
    if (offset_info.size != length_info.size ||
        offset_info.size != sample_index_info.size ||
        offset_info.size != segment_index_info.size) {
        throw std::invalid_argument(
            "offsets, lengths, sample_indices, and segment_indices must have the same size");
    }

    const auto* sample_data = static_cast<const std::complex<float>*>(sample_info.ptr);
    const auto* offset_data = static_cast<const std::int64_t*>(offset_info.ptr);
    const auto* length_data = static_cast<const std::int64_t*>(length_info.ptr);
    const auto* sample_index_data = static_cast<const std::int64_t*>(sample_index_info.ptr);
    const auto* segment_index_data = static_cast<const std::int64_t*>(segment_index_info.ptr);
    const auto* tap_data = static_cast<const double*>(tap_info.ptr);

    std::vector<bt_native::BrPacketCandidate> candidates;
    {
        py::gil_scoped_release release;
        candidates = bt_native::build_br_packet_candidates(
            sample_data,
            static_cast<std::int64_t>(sample_info.size),
            offset_data,
            length_data,
            sample_index_data,
            segment_index_data,
            static_cast<std::int64_t>(offset_info.size),
            tap_data,
            static_cast<std::int64_t>(tap_info.size),
            gain,
            samples_per_bit,
            sample_rate,
            thread_count);
    }

    py::list output;
    for (const auto& candidate : candidates) {
        py::dict item;
        item["sample_index"] = candidate.sample_index;
        item["segment_index"] = candidate.segment_index;
        item["lap"] = candidate.lap;
        item["rssi"] = candidate.rssi;
        item["cfo_hz"] = candidate.cfo_hz.has_value() ? py::cast(*candidate.cfo_hz) : py::none();
        item["raw_bits_len"] = candidate.raw_bits_len;
        py::list payload_bits;
        for (const auto bit : candidate.payload_bits) {
            payload_bits.append(bit);
        }
        item["payload_bits"] = payload_bits;
        py::list header_candidates;
        for (const auto& header : candidate.header_candidates) {
            py::dict header_item;
            header_item["uap"] = header.uap;
            header_item["clk"] = header.clk;
            header_item["header"] = header.header;
            header_item["lfsr"] = header.lfsr;
            header_item["payload_bit_count"] = header.payload_bit_count;
            header_item["payload_prefix_bits"] = header.payload_prefix_bits;
            header_item["payload_prefix_count"] = header.payload_prefix_count;
            header_candidates.append(header_item);
        }
        item["header_candidates"] = header_candidates;
        output.append(item);
    }
    return output;
}

py::list br_candidates_to_python(const std::vector<bt_native::BrPacketCandidate>& candidates)
{
    py::list output;
    for (const auto& candidate : candidates) {
        py::dict item;
        item["sample_index"] = candidate.sample_index;
        item["segment_index"] = candidate.segment_index;
        item["lap"] = candidate.lap;
        item["rssi"] = candidate.rssi;
        item["cfo_hz"] = candidate.cfo_hz.has_value() ? py::cast(*candidate.cfo_hz) : py::none();
        item["raw_bits_len"] = candidate.raw_bits_len;
        py::list payload_bits;
        for (const auto bit : candidate.payload_bits) {
            payload_bits.append(bit);
        }
        item["payload_bits"] = payload_bits;
        py::list header_candidates;
        for (const auto& header : candidate.header_candidates) {
            py::dict header_item;
            header_item["uap"] = header.uap;
            header_item["clk"] = header.clk;
            header_item["header"] = header.header;
            header_item["lfsr"] = header.lfsr;
            header_item["payload_bit_count"] = header.payload_bit_count;
            header_item["payload_prefix_bits"] = header.payload_prefix_bits;
            header_item["payload_prefix_count"] = header.payload_prefix_count;
            header_candidates.append(header_item);
        }
        item["header_candidates"] = header_candidates;
        output.append(item);
    }
    return output;
}

py::list build_br_packet_candidates_buffers_complex64(
    py::list buffers,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> buffer_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> buffer_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sample_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> segment_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> lpf_taps,
    double gain,
    int samples_per_bit,
    double sample_rate,
    int thread_count)
{
    const auto packed = pack_segment_buffers_complex64(
        buffers,
        buffer_indices,
        buffer_offsets,
        lengths,
        lengths,
        sample_indices,
        segment_indices);
    const py::buffer_info tap_info = lpf_taps.request();
    const auto* tap_data = static_cast<const double*>(tap_info.ptr);
    std::vector<bt_native::BrPacketCandidate> candidates;
    {
        py::gil_scoped_release release;
        candidates = bt_native::build_br_packet_candidates(
            packed.samples.data(),
            static_cast<std::int64_t>(packed.samples.size()),
            packed.offsets.data(),
            packed.lengths.data(),
            packed.sample_indices.data(),
            packed.segment_indices.data(),
            static_cast<std::int64_t>(packed.offsets.size()),
            tap_data,
            static_cast<std::int64_t>(tap_info.size),
            gain,
            samples_per_bit,
            sample_rate,
            thread_count);
    }
    return br_candidates_to_python(candidates);
}

py::dict build_and_process_br_packets_complex64(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> samples,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sample_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> segment_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> lpf_taps,
    double gain,
    int samples_per_bit,
    double sample_rate,
    py::object locked_uap,
    int miss_count,
    int max_miss,
    bool retry_after_unlock,
    int thread_count)
{
    const py::buffer_info sample_info = samples.request();
    const py::buffer_info offset_info = offsets.request();
    const py::buffer_info length_info = lengths.request();
    const py::buffer_info sample_index_info = sample_indices.request();
    const py::buffer_info segment_index_info = segment_indices.request();
    const py::buffer_info tap_info = lpf_taps.request();
    if (offset_info.size != length_info.size ||
        offset_info.size != sample_index_info.size ||
        offset_info.size != segment_index_info.size) {
        throw std::invalid_argument(
            "offsets, lengths, sample_indices, and segment_indices must have the same size");
    }

    const auto* sample_data = static_cast<const std::complex<float>*>(sample_info.ptr);
    const auto* offset_data = static_cast<const std::int64_t*>(offset_info.ptr);
    const auto* length_data = static_cast<const std::int64_t*>(length_info.ptr);
    const auto* sample_index_data = static_cast<const std::int64_t*>(sample_index_info.ptr);
    const auto* segment_index_data = static_cast<const std::int64_t*>(segment_index_info.ptr);
    const auto* tap_data = static_cast<const double*>(tap_info.ptr);
    const auto candidates = bt_native::build_br_packet_candidates(
        sample_data,
        static_cast<std::int64_t>(sample_info.size),
        offset_data,
        length_data,
        sample_index_data,
        segment_index_data,
        static_cast<std::int64_t>(offset_info.size),
        tap_data,
        static_cast<std::int64_t>(tap_info.size),
        gain,
        samples_per_bit,
        sample_rate,
        thread_count);

    std::vector<std::vector<bt_native::BrHeaderCandidate>> candidate_batches;
    std::vector<std::int64_t> raw_bits_lens;
    candidate_batches.reserve(candidates.size());
    raw_bits_lens.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        candidate_batches.push_back(candidate.header_candidates);
        raw_bits_lens.push_back(candidate.raw_bits_len);
    }
    const auto decisions = bt_native::process_br_header_candidate_sequence(
        candidate_batches,
        raw_bits_lens,
        optional_int_from_python(locked_uap),
        miss_count,
        max_miss,
        retry_after_unlock);

    py::list packets;
    for (std::size_t idx = 0; idx < candidates.size(); ++idx) {
        const auto& candidate = candidates[idx];
        const auto& decision = decisions[idx];
        py::dict item;
        item["sample_index"] = candidate.sample_index;
        item["segment_index"] = candidate.segment_index;
        item["lap"] = candidate.lap;
        item["rssi"] = candidate.rssi;
        item["cfo_hz"] = candidate.cfo_hz.has_value() ? py::cast(*candidate.cfo_hz) : py::none();
        item["status"] = decision.status;
        item["uap"] = decision.hec_ok ? py::cast(decision.uap) : py::cast("");
        item["clk"] = decision.clk;
        item["type"] = decision.hec_ok ? py::cast(decision.type_name) : py::cast("");
        item["type_val"] = decision.hec_ok ? py::cast(decision.type_val) : py::cast("");
        item["len"] = decision.hec_ok ? py::cast(decision.length) : py::cast("");
        item["total_bytes"] = decision.hec_ok
            ? (decision.total_bytes >= 0 ? py::cast(decision.total_bytes) : py::none())
            : py::cast("");
        item["hec_ok"] = decision.hec_ok;
        packets.append(item);
    }

    py::dict output;
    output["packets"] = packets;
    if (!decisions.empty()) {
        const auto& last = decisions.back();
        output["locked_uap"] = last.locked_uap.has_value() ? py::cast(*last.locked_uap) : py::none();
        output["miss_count"] = last.miss_count;
    } else {
        output["locked_uap"] = locked_uap.is_none() ? py::none() : locked_uap;
        output["miss_count"] = miss_count;
    }
    return output;
}

py::dict process_br_candidates_to_packet_dict(
    const std::vector<bt_native::BrPacketCandidate>& candidates,
    py::object locked_uap,
    int miss_count,
    int max_miss,
    bool retry_after_unlock)
{
    std::vector<std::vector<bt_native::BrHeaderCandidate>> candidate_batches;
    std::vector<std::int64_t> raw_bits_lens;
    candidate_batches.reserve(candidates.size());
    raw_bits_lens.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        candidate_batches.push_back(candidate.header_candidates);
        raw_bits_lens.push_back(candidate.raw_bits_len);
    }
    const auto decisions = bt_native::process_br_header_candidate_sequence(
        candidate_batches,
        raw_bits_lens,
        optional_int_from_python(locked_uap),
        miss_count,
        max_miss,
        retry_after_unlock);

    py::list packets;
    for (std::size_t idx = 0; idx < candidates.size(); ++idx) {
        const auto& candidate = candidates[idx];
        const auto& decision = decisions[idx];
        py::dict item;
        item["sample_index"] = candidate.sample_index;
        item["segment_index"] = candidate.segment_index;
        item["lap"] = candidate.lap;
        item["rssi"] = candidate.rssi;
        item["cfo_hz"] = candidate.cfo_hz.has_value() ? py::cast(*candidate.cfo_hz) : py::none();
        item["status"] = decision.status;
        item["uap"] = decision.hec_ok ? py::cast(decision.uap) : py::cast("");
        item["clk"] = decision.clk;
        item["type"] = decision.hec_ok ? py::cast(decision.type_name) : py::cast("");
        item["type_val"] = decision.hec_ok ? py::cast(decision.type_val) : py::cast("");
        item["len"] = decision.hec_ok ? py::cast(decision.length) : py::cast("");
        item["total_bytes"] = decision.hec_ok
            ? (decision.total_bytes >= 0 ? py::cast(decision.total_bytes) : py::none())
            : py::cast("");
        item["hec_ok"] = decision.hec_ok;
        packets.append(item);
    }

    py::dict output;
    output["packets"] = packets;
    if (!decisions.empty()) {
        const auto& last = decisions.back();
        output["locked_uap"] = last.locked_uap.has_value() ? py::cast(*last.locked_uap) : py::none();
        output["miss_count"] = last.miss_count;
    } else {
        output["locked_uap"] = locked_uap.is_none() ? py::none() : locked_uap;
        output["miss_count"] = miss_count;
    }
    return output;
}

py::dict build_and_process_br_packets_buffers_complex64(
    py::list buffers,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> buffer_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> buffer_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sample_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> segment_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> lpf_taps,
    double gain,
    int samples_per_bit,
    double sample_rate,
    py::object locked_uap,
    int miss_count,
    int max_miss,
    bool retry_after_unlock,
    int thread_count)
{
    const auto packed = pack_segment_buffers_complex64(
        buffers,
        buffer_indices,
        buffer_offsets,
        lengths,
        lengths,
        sample_indices,
        segment_indices);
    const py::buffer_info tap_info = lpf_taps.request();
    const auto* tap_data = static_cast<const double*>(tap_info.ptr);
    const auto candidates = bt_native::build_br_packet_candidates(
        packed.samples.data(),
        static_cast<std::int64_t>(packed.samples.size()),
        packed.offsets.data(),
        packed.lengths.data(),
        packed.sample_indices.data(),
        packed.segment_indices.data(),
        static_cast<std::int64_t>(packed.offsets.size()),
        tap_data,
        static_cast<std::int64_t>(tap_info.size),
        gain,
        samples_per_bit,
        sample_rate,
        thread_count);
    return process_br_candidates_to_packet_dict(
        candidates,
        locked_uap,
        miss_count,
        max_miss,
        retry_after_unlock);
}

py::list decode_br_header_candidates_bits(
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> raw_bits,
    py::array_t<int, py::array::c_style | py::array::forcecast> candidate_uaps,
    bool stop_after_first)
{
    const py::buffer_info bit_info = raw_bits.request();
    const py::buffer_info uap_info = candidate_uaps.request();
    const auto* bit_data = static_cast<const std::uint8_t*>(bit_info.ptr);
    const auto* uap_data = static_cast<const int*>(uap_info.ptr);
    const auto candidates = bt_native::decode_br_header_candidates(
        bit_data,
        static_cast<std::int64_t>(bit_info.size),
        uap_data,
        static_cast<std::int64_t>(uap_info.size),
        stop_after_first);

    py::list output;
    for (const auto& candidate : candidates) {
        py::dict item;
        item["uap"] = candidate.uap;
        item["clk"] = candidate.clk;
        item["header"] = candidate.header;
        item["lfsr"] = candidate.lfsr;
        item["payload_bit_count"] = candidate.payload_bit_count;
        item["payload_prefix_bits"] = candidate.payload_prefix_bits;
        item["payload_prefix_count"] = candidate.payload_prefix_count;
        output.append(item);
    }
    return output;
}

bt_native::BrHeaderCandidate br_header_candidate_from_dict(const py::dict& item)
{
    return {
        py::cast<int>(item["uap"]),
        py::cast<int>(item["clk"]),
        py::cast<std::uint32_t>(item["header"]),
        py::cast<int>(item["lfsr"]),
        item.contains("payload_bit_count")
            ? py::cast<std::int64_t>(item["payload_bit_count"])
            : (item.contains("payload_len") ? py::cast<std::int64_t>(item["payload_len"]) : 0),
        item.contains("payload_prefix_bits")
            ? py::cast<std::uint16_t>(item["payload_prefix_bits"])
            : static_cast<std::uint16_t>(0),
        item.contains("payload_prefix_count")
            ? py::cast<int>(item["payload_prefix_count"])
            : 0,
    };
}

std::vector<bt_native::BrHeaderCandidate> br_header_candidates_from_list(py::list candidates)
{
    std::vector<bt_native::BrHeaderCandidate> native_candidates;
    native_candidates.reserve(static_cast<std::size_t>(py::len(candidates)));
    for (const py::handle item_handle : candidates) {
        native_candidates.push_back(
            br_header_candidate_from_dict(py::reinterpret_borrow<py::dict>(item_handle)));
    }
    return native_candidates;
}

std::optional<int> optional_int_from_python(py::object value)
{
    std::optional<int> native_locked_uap;
    if (!value.is_none()) {
        native_locked_uap = py::cast<int>(value);
    }
    return native_locked_uap;
}

py::dict br_packet_decision_to_dict(const bt_native::BrPacketDecision& result)
{
    py::dict output;
    output["status"] = result.status;
    output["uap"] = result.hec_ok ? py::cast(result.uap) : py::cast("");
    output["clk"] = result.clk;
    output["type"] = result.hec_ok ? py::cast(result.type_name) : py::cast("");
    output["type_val"] = result.hec_ok ? py::cast(result.type_val) : py::cast("");
    output["len"] = result.hec_ok ? py::cast(result.length) : py::cast("");
    output["total_bytes"] = result.hec_ok
        ? (result.total_bytes >= 0 ? py::cast(result.total_bytes) : py::none())
        : py::cast("");
    output["locked_uap"] = result.locked_uap.has_value() ? py::cast(*result.locked_uap) : py::none();
    output["miss_count"] = result.miss_count;
    output["hec_ok"] = result.hec_ok;
    return output;
}

py::dict process_br_header_candidates(
    py::list candidates,
    std::int64_t raw_bits_len,
    py::object locked_uap,
    int miss_count,
    int max_miss,
    bool retry_after_unlock)
{
    const auto native_candidates = br_header_candidates_from_list(candidates);
    const auto result = bt_native::process_br_header_candidates(
        native_candidates,
        raw_bits_len,
        optional_int_from_python(locked_uap),
        miss_count,
        max_miss,
        retry_after_unlock);
    return br_packet_decision_to_dict(result);
}

py::list process_br_header_candidate_sequence(
    py::list candidate_batches,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> raw_bits_lens,
    py::object locked_uap,
    int miss_count,
    int max_miss,
    bool retry_after_unlock)
{
    const py::buffer_info raw_len_info = raw_bits_lens.request();
    if (static_cast<std::size_t>(raw_len_info.size) != static_cast<std::size_t>(py::len(candidate_batches))) {
        throw std::invalid_argument("candidate_batches and raw_bits_lens must have the same size");
    }
    const auto* raw_len_data = static_cast<const std::int64_t*>(raw_len_info.ptr);

    std::vector<std::vector<bt_native::BrHeaderCandidate>> native_batches;
    std::vector<std::int64_t> native_raw_lens;
    native_batches.reserve(static_cast<std::size_t>(py::len(candidate_batches)));
    native_raw_lens.reserve(static_cast<std::size_t>(raw_len_info.size));
    for (const py::handle batch_handle : candidate_batches) {
        native_batches.push_back(br_header_candidates_from_list(py::reinterpret_borrow<py::list>(batch_handle)));
    }
    for (std::size_t idx = 0; idx < static_cast<std::size_t>(raw_len_info.size); ++idx) {
        native_raw_lens.push_back(raw_len_data[idx]);
    }

    const auto decisions = bt_native::process_br_header_candidate_sequence(
        native_batches,
        native_raw_lens,
        optional_int_from_python(locked_uap),
        miss_count,
        max_miss,
        retry_after_unlock);

    py::list output;
    for (const auto& decision : decisions) {
        output.append(br_packet_decision_to_dict(decision));
    }
    return output;
}

double compute_rssi_db_complex64(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> samples,
    double gain_offset_db)
{
    const py::buffer_info info = samples.request();
    const auto* data = static_cast<const std::complex<float>*>(info.ptr);
    return bt_native::compute_rssi_db(data, static_cast<std::int64_t>(info.size), gain_offset_db);
}

py::list summarize_segments_complex64(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> samples,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths)
{
    const py::buffer_info sample_info = samples.request();
    const py::buffer_info offset_info = offsets.request();
    const py::buffer_info length_info = lengths.request();
    if (offset_info.size != length_info.size) {
        throw std::invalid_argument("offsets and lengths must have the same size");
    }

    const auto* sample_data = static_cast<const std::complex<float>*>(sample_info.ptr);
    const auto* offset_data = static_cast<const std::int64_t*>(offset_info.ptr);
    const auto* length_data = static_cast<const std::int64_t*>(length_info.ptr);
    const auto summaries = bt_native::summarize_segments(
        sample_data,
        static_cast<std::int64_t>(sample_info.size),
        offset_data,
        length_data,
        static_cast<std::int64_t>(offset_info.size));

    py::list output;
    for (const auto& summary : summaries) {
        py::dict item;
        item["offset"] = summary.offset;
        item["length"] = summary.length;
        item["power_sum"] = summary.power_sum;
        output.append(item);
    }
    return output;
}

}  // namespace

PYBIND11_MODULE(bt_native, m)
{
    m.doc() = "Experimental native parser extension for Bluetooth capture analysis";
    m.def("version", &bt_native::version);
    m.def("self_test", &bt_native::self_test);
    m.def("sum_float32", &sum_float32, py::arg("values"));
    m.def("gfsk_demodulate_complex64", &gfsk_demodulate_complex64, py::arg("samples"), py::arg("gain"));
    m.def("decision_bits_float64", &decision_bits_float64, py::arg("freq_dev"), py::arg("samples_per_symbol"));
    m.def(
        "detect_ble_access_address_bits",
        &detect_ble_access_address_bits,
        py::arg("bits"),
        py::arg("ble_mode"));
    m.def(
        "valid_ble_connection_access_address",
        &bt_native::valid_ble_connection_access_address,
        py::arg("access_address_raw"));
    m.def(
        "parse_ble_packet_bits",
        &parse_ble_packet_bits,
        py::arg("bits"),
        py::arg("ble_mode"),
        py::arg("iq_len"),
        py::arg("channel"));
    m.def(
        "parse_ble_segments_complex64",
        &parse_ble_segments_complex64,
        py::arg("samples"),
        py::arg("offsets"),
        py::arg("lengths"),
        py::arg("score_lengths"),
        py::arg("sample_indices"),
        py::arg("gain"),
        py::arg("score_threshold"),
        py::arg("channel"),
        py::arg("thread_count") = 1);
    m.def(
        "parse_ble_segment_buffers_complex64",
        &parse_ble_segment_buffers_complex64,
        py::arg("buffers"),
        py::arg("buffer_indices"),
        py::arg("offsets"),
        py::arg("lengths"),
        py::arg("score_lengths"),
        py::arg("sample_indices"),
        py::arg("segment_indices"),
        py::arg("gain"),
        py::arg("score_threshold"),
        py::arg("channel"),
        py::arg("thread_count") = 1);
    m.def("build_bluetooth_sync_word", &build_bluetooth_sync_word, py::arg("lap"));
    m.def("find_br_access_code_bits", &find_br_access_code_bits, py::arg("bits"));
    m.def(
        "build_br_packet_candidates_complex64",
        &build_br_packet_candidates_complex64,
        py::arg("samples"),
        py::arg("offsets"),
        py::arg("lengths"),
        py::arg("sample_indices"),
        py::arg("segment_indices"),
        py::arg("lpf_taps"),
        py::arg("gain"),
        py::arg("samples_per_bit"),
        py::arg("sample_rate"),
        py::arg("thread_count") = 1);
    m.def(
        "build_br_packet_candidates_buffers_complex64",
        &build_br_packet_candidates_buffers_complex64,
        py::arg("buffers"),
        py::arg("buffer_indices"),
        py::arg("offsets"),
        py::arg("lengths"),
        py::arg("sample_indices"),
        py::arg("segment_indices"),
        py::arg("lpf_taps"),
        py::arg("gain"),
        py::arg("samples_per_bit"),
        py::arg("sample_rate"),
        py::arg("thread_count") = 1);
    m.def(
        "build_and_process_br_packets_complex64",
        &build_and_process_br_packets_complex64,
        py::arg("samples"),
        py::arg("offsets"),
        py::arg("lengths"),
        py::arg("sample_indices"),
        py::arg("segment_indices"),
        py::arg("lpf_taps"),
        py::arg("gain"),
        py::arg("samples_per_bit"),
        py::arg("sample_rate"),
        py::arg("locked_uap") = py::none(),
        py::arg("miss_count") = 0,
        py::arg("max_miss") = 10,
        py::arg("retry_after_unlock") = false,
        py::arg("thread_count") = 1);
    m.def(
        "build_and_process_br_packets_buffers_complex64",
        &build_and_process_br_packets_buffers_complex64,
        py::arg("buffers"),
        py::arg("buffer_indices"),
        py::arg("offsets"),
        py::arg("lengths"),
        py::arg("sample_indices"),
        py::arg("segment_indices"),
        py::arg("lpf_taps"),
        py::arg("gain"),
        py::arg("samples_per_bit"),
        py::arg("sample_rate"),
        py::arg("locked_uap") = py::none(),
        py::arg("miss_count") = 0,
        py::arg("max_miss") = 10,
        py::arg("retry_after_unlock") = false,
        py::arg("thread_count") = 1);
    m.def(
        "decode_br_header_candidates_bits",
        &decode_br_header_candidates_bits,
        py::arg("raw_bits"),
        py::arg("candidate_uaps"),
        py::arg("stop_after_first") = false);
    m.def(
        "process_br_header_candidates",
        &process_br_header_candidates,
        py::arg("candidates"),
        py::arg("raw_bits_len"),
        py::arg("locked_uap") = py::none(),
        py::arg("miss_count") = 0,
        py::arg("max_miss") = 10,
        py::arg("retry_after_unlock") = true);
    m.def(
        "process_br_header_candidate_sequence",
        &process_br_header_candidate_sequence,
        py::arg("candidate_batches"),
        py::arg("raw_bits_lens"),
        py::arg("locked_uap") = py::none(),
        py::arg("miss_count") = 0,
        py::arg("max_miss") = 10,
        py::arg("retry_after_unlock") = true);
    m.def(
        "compute_rssi_db_complex64",
        &compute_rssi_db_complex64,
        py::arg("samples"),
        py::arg("gain_offset_db") = -30.0);
    m.def(
        "summarize_segments_complex64",
        &summarize_segments_complex64,
        py::arg("samples"),
        py::arg("offsets"),
        py::arg("lengths"));
}
