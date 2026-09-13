#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
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

struct CovertFrameFields {
    bool found = false;
    bool integrity_ok = false;
    std::uint16_t seq = 0;
    std::vector<std::uint8_t> payload;
    std::vector<std::uint8_t> frame;
};

std::string format_hex(const std::vector<std::uint8_t>& bytes)
{
    static constexpr char digits[] = "0123456789ABCDEF";
    std::string output;
    output.reserve(bytes.size() * 2);
    for (const auto value : bytes) {
        output.push_back(digits[(value >> 4) & 0x0F]);
        output.push_back(digits[value & 0x0F]);
    }
    return output;
}

int hex_digit(char value)
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
    return -1;
}

std::vector<std::uint8_t> parse_hex(const std::string& text)
{
    if (text.size() % 2 != 0) {
        return {};
    }
    std::vector<std::uint8_t> bytes;
    bytes.reserve(text.size() / 2);
    for (std::size_t index = 0; index < text.size(); index += 2) {
        const int high = hex_digit(text[index]);
        const int low = hex_digit(text[index + 1]);
        if (high < 0 || low < 0) {
            return {};
        }
        bytes.push_back(static_cast<std::uint8_t>((high << 4) | low));
    }
    return bytes;
}

CovertFrameFields decode_phantom_frame(const std::string& post_crc_hex)
{
    const auto bytes = parse_hex(post_crc_hex);
    CovertFrameFields first_candidate;
    for (std::size_t start = 0; start + 6 <= bytes.size(); ++start) {
        if (bytes[start] != 0x50 || bytes[start + 1] != 0x43) {
            continue;
        }
        const auto covert_len = static_cast<std::size_t>(bytes[start + 4]);
        const auto frame_end = start + 5 + covert_len + 1;
        if (frame_end > bytes.size()) {
            continue;
        }

        CovertFrameFields candidate;
        candidate.found = true;
        candidate.seq = static_cast<std::uint16_t>(bytes[start + 2]) |
            (static_cast<std::uint16_t>(bytes[start + 3]) << 8);
        candidate.frame.assign(bytes.begin() + static_cast<std::ptrdiff_t>(start),
            bytes.begin() + static_cast<std::ptrdiff_t>(frame_end));
        candidate.payload.assign(
            bytes.begin() + static_cast<std::ptrdiff_t>(start + 5),
            bytes.begin() + static_cast<std::ptrdiff_t>(start + 5 + covert_len));

        std::uint8_t checksum = 0;
        for (std::size_t index = start; index + 1 < frame_end; ++index) {
            checksum ^= bytes[index];
        }
        candidate.integrity_ok = checksum == bytes[frame_end - 1];
        if (candidate.integrity_ok) {
            return candidate;
        }
        if (!first_candidate.found) {
            first_candidate = std::move(candidate);
        }
    }
    return first_candidate;
}

py::dict candidate_to_dict(const bt_native::BlePacketCandidate& candidate)
{
    py::dict output;
    py::list access_address;
    for (std::size_t index = 0; index < candidate.access_address.size(); index += 2) {
        access_address.append("0x" + candidate.access_address.substr(index, 2));
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

    const auto covert = decode_phantom_frame(candidate.post_crc_hex);
    output["covert_len"] = covert.found ? static_cast<int>(covert.payload.size()) : 0;
    output["covert_data_len"] = 0;
    output["covert_hex"] = "";
    output["covert_data"] = "";
    output["covert_data_hex"] = "";
    output["covert_marker_hex"] = "";
    output["covert_frame_hex"] = "";
    output["covert_seq"] = "";
    output["covert_integrity_ok"] = "";
    if (covert.found) {
        output["covert_hex"] = format_hex(covert.payload);
        output["covert_frame_hex"] = format_hex(covert.frame);
        output["covert_seq"] = covert.seq;
        output["covert_integrity_ok"] = covert.integrity_ok ? "1" : "0";
        if (!covert.payload.empty()) {
            const std::vector<std::uint8_t> marker(covert.payload.begin(), covert.payload.begin() + 1);
            output["covert_marker_hex"] = format_hex(marker);
            const std::vector<std::uint8_t> data(covert.payload.begin() + 1, covert.payload.end());
            output["covert_data_len"] = static_cast<int>(data.size());
            output["covert_data"] = format_hex(data);
            output["covert_data_hex"] = format_hex(data);
        }
    }
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

PackedSegmentInput pack_segment_buffers(
    py::list buffers,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> buffer_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> buffer_offsets,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> score_lengths,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> sample_indices,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> segment_indices)
{
    const auto buffer_index_info = buffer_indices.request();
    const auto offset_info = buffer_offsets.request();
    const auto length_info = lengths.request();
    const auto score_info = score_lengths.request();
    const auto sample_info = sample_indices.request();
    const auto segment_info = segment_indices.request();
    if (buffer_index_info.size != offset_info.size ||
        offset_info.size != length_info.size ||
        length_info.size != score_info.size ||
        score_info.size != sample_info.size ||
        sample_info.size != segment_info.size) {
        throw std::invalid_argument("segment descriptor arrays must have the same size");
    }

    PackedSegmentInput packed;
    const auto segment_count = static_cast<std::size_t>(buffer_index_info.size);
    packed.buffers.reserve(static_cast<std::size_t>(py::len(buffers)));
    std::vector<py::buffer_info> infos;
    infos.reserve(static_cast<std::size_t>(py::len(buffers)));
    for (const py::handle item : buffers) {
        packed.buffers.emplace_back(py::reinterpret_borrow<py::object>(item));
        infos.push_back(packed.buffers.back().request());
        if (infos.back().ndim != 1) {
            throw std::invalid_argument("each segment buffer must be one-dimensional");
        }
    }

    const auto* buffer_index_data = static_cast<const std::int64_t*>(buffer_index_info.ptr);
    const auto* offset_data = static_cast<const std::int64_t*>(offset_info.ptr);
    const auto* length_data = static_cast<const std::int64_t*>(length_info.ptr);
    const auto* score_data = static_cast<const std::int64_t*>(score_info.ptr);
    const auto* sample_data = static_cast<const std::int64_t*>(sample_info.ptr);
    const auto* segment_data = static_cast<const std::int64_t*>(segment_info.ptr);

    std::int64_t total_samples = 0;
    packed.offsets.reserve(segment_count);
    packed.lengths.reserve(segment_count);
    packed.score_lengths.reserve(segment_count);
    packed.sample_indices.reserve(segment_count);
    packed.segment_indices.reserve(segment_count);
    for (std::size_t index = 0; index < segment_count; ++index) {
        const auto buffer_index = buffer_index_data[index];
        const auto offset = offset_data[index];
        const auto length = length_data[index];
        if (buffer_index < 0 || static_cast<std::size_t>(buffer_index) >= infos.size()) {
            throw std::out_of_range("segment buffer index is out of range");
        }
        if (offset < 0 || length < 0 || offset + length > infos[static_cast<std::size_t>(buffer_index)].size) {
            throw std::out_of_range("segment offset/length exceeds buffer bounds");
        }
        packed.offsets.push_back(total_samples);
        packed.lengths.push_back(length);
        packed.score_lengths.push_back(score_data[index]);
        packed.sample_indices.push_back(sample_data[index]);
        packed.segment_indices.push_back(segment_data[index]);
        total_samples += length;
    }

    packed.samples.resize(static_cast<std::size_t>(total_samples));
    for (std::size_t index = 0; index < segment_count; ++index) {
        const auto buffer_index = static_cast<std::size_t>(buffer_index_data[index]);
        const auto offset = static_cast<std::size_t>(offset_data[index]);
        const auto length = static_cast<std::size_t>(length_data[index]);
        const auto output_offset = static_cast<std::size_t>(packed.offsets[index]);
        const auto* source = static_cast<const std::complex<float>*>(infos[buffer_index].ptr);
        std::copy(source + offset, source + offset + length, packed.samples.data() + output_offset);
    }
    return packed;
}

py::list parse_segments(
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
    const auto packets = bt_native::parse_ble_segments(
        samples, sample_count, offsets, lengths, score_lengths, sample_indices,
        segment_count, gain, score_threshold, channel, thread_count);
    py::list output;
    for (const auto& packet : packets) {
        output.append(candidate_to_dict(packet));
    }
    return output;
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
    const auto sample_info = samples.request();
    const auto offset_info = offsets.request();
    const auto length_info = lengths.request();
    const auto score_info = score_lengths.request();
    const auto index_info = sample_indices.request();
    if (offset_info.size != length_info.size ||
        length_info.size != score_info.size ||
        score_info.size != index_info.size) {
        throw std::invalid_argument("segment descriptor arrays must have the same size");
    }
    return parse_segments(
        static_cast<const std::complex<float>*>(sample_info.ptr),
        static_cast<std::int64_t>(sample_info.size),
        static_cast<const std::int64_t*>(offset_info.ptr),
        static_cast<const std::int64_t*>(length_info.ptr),
        static_cast<const std::int64_t*>(score_info.ptr),
        static_cast<const std::int64_t*>(index_info.ptr),
        static_cast<std::int64_t>(offset_info.size),
        gain, score_threshold, channel, thread_count);
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
    const auto packed = pack_segment_buffers(
        buffers, buffer_indices, buffer_offsets, lengths, score_lengths,
        sample_indices, segment_indices);
    return parse_segments(
        packed.samples.data(), static_cast<std::int64_t>(packed.samples.size()),
        packed.offsets.data(), packed.lengths.data(), packed.score_lengths.data(),
        packed.sample_indices.data(), static_cast<std::int64_t>(packed.offsets.size()),
        gain, score_threshold, channel, thread_count);
}

}  // namespace

PYBIND11_MODULE(bt_native, module)
{
    module.doc() = "PhantomChannel-local BLE native facade";
    module.def("version", &bt_native::version);
    module.def("self_test", &bt_native::self_test);
    module.def(
        "parse_ble_segments_complex64",
        &parse_ble_segments_complex64,
        py::arg("samples"), py::arg("offsets"), py::arg("lengths"),
        py::arg("score_lengths"), py::arg("sample_indices"), py::arg("gain"),
        py::arg("score_threshold"), py::arg("channel"), py::arg("thread_count") = 1);
    module.def(
        "parse_ble_segment_buffers_complex64",
        &parse_ble_segment_buffers_complex64,
        py::arg("buffers"), py::arg("buffer_indices"), py::arg("buffer_offsets"),
        py::arg("lengths"), py::arg("score_lengths"), py::arg("sample_indices"),
        py::arg("segment_indices"), py::arg("gain"), py::arg("score_threshold"),
        py::arg("channel"), py::arg("thread_count") = 1);
}
