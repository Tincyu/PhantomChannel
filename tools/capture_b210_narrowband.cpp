// Minimal B210 narrowband SC16 capture for the phone-HRS hardware validation.
// This intentionally sets the analog RX bandwidth explicitly; the legacy
// framed B210 capture binary only follows the sample rate and may leave the
// frontend at a much wider bandwidth.

#include <uhd/types/metadata.hpp>
#include <uhd/types/stream_cmd.hpp>
#include <uhd/utils/thread.hpp>
#include <uhd/usrp/multi_usrp.hpp>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    std::string usrp_args = "type=b200,master_clock_rate=16e6";
    std::string output;
    std::string metadata;
    double rate = 2e6;
    double bandwidth = 2e6;
    double frequency = 2440e6;
    double gain = 50.0;
    std::string antenna = "RX2";
    size_t channel = 0;
    double duration = 10.0;
    size_t block_samples = 4096;
};

static std::string value_after(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

static Options parse_options(int argc, char** argv) {
    Options o;
    o.usrp_args = value_after(argc, argv, "--args", o.usrp_args);
    o.output = value_after(argc, argv, "--output", o.output);
    o.metadata = value_after(argc, argv, "--metadata", o.metadata);
    o.rate = std::stod(value_after(argc, argv, "--rate", std::to_string(o.rate)));
    o.bandwidth = std::stod(value_after(argc, argv, "--bandwidth", std::to_string(o.bandwidth)));
    o.frequency = std::stod(value_after(argc, argv, "--frequency", std::to_string(o.frequency)));
    o.gain = std::stod(value_after(argc, argv, "--gain", std::to_string(o.gain)));
    o.antenna = value_after(argc, argv, "--antenna", o.antenna);
    o.channel = static_cast<size_t>(std::stoul(value_after(argc, argv, "--channel", std::to_string(o.channel))));
    o.duration = std::stod(value_after(argc, argv, "--duration", std::to_string(o.duration)));
    o.block_samples = static_cast<size_t>(std::stoul(value_after(argc, argv, "--block-samples", std::to_string(o.block_samples))));
    if (o.output.empty() || o.metadata.empty()) {
        throw std::runtime_error("required: --output <iq.sc16> --metadata <metadata.json>");
    }
    return o;
}

static void write_metadata(
    const Options& o,
    const uhd::usrp::multi_usrp::sptr& usrp,
    size_t samples,
    size_t overflows,
    size_t timeouts,
    size_t other_errors,
    const std::string& status) {
    std::ofstream out(o.metadata);
    if (!out) {
        throw std::runtime_error("cannot write metadata: " + o.metadata);
    }
    out << std::setprecision(17);
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"source\": \"usrp_b210\",\n"
        << "  \"device_model\": \"B210\",\n"
        << "  \"usrp_args\": " << std::quoted(o.usrp_args) << ",\n"
        << "  \"channel\": " << o.channel << ",\n"
        << "  \"antenna\": " << std::quoted(o.antenna) << ",\n"
        << "  \"requested_sample_rate_sps\": " << o.rate << ",\n"
        << "  \"actual_sample_rate_sps\": " << usrp->get_rx_rate(o.channel) << ",\n"
        << "  \"requested_rx_bandwidth_hz\": " << o.bandwidth << ",\n"
        << "  \"actual_rx_bandwidth_hz\": " << usrp->get_rx_bandwidth(o.channel) << ",\n"
        << "  \"requested_center_frequency_hz\": " << o.frequency << ",\n"
        << "  \"actual_center_frequency_hz\": " << usrp->get_rx_freq(o.channel) << ",\n"
        << "  \"requested_gain_db\": " << o.gain << ",\n"
        << "  \"actual_gain_db\": " << usrp->get_rx_gain(o.channel) << ",\n"
        << "  \"duration_s\": " << o.duration << ",\n"
        << "  \"samples\": " << samples << ",\n"
        << "  \"bytes_per_complex_sample\": 4,\n"
        << "  \"overflows\": " << overflows << ",\n"
        << "  \"timeouts\": " << timeouts << ",\n"
        << "  \"other_errors\": " << other_errors << ",\n"
        << "  \"status\": " << std::quoted(status) << "\n"
        << "}\n";
}

int main(int argc, char** argv) {
    try {
        const Options o = parse_options(argc, argv);
        // Keep the host receive loop responsive at the fixed 4 MS/s rate.
        // This is best-effort: the capture remains usable on systems where
        // realtime scheduling is not permitted.
        uhd::set_thread_priority_safe(uhd::DEFAULT_THREAD_PRIORITY, true);
        std::cout << "Creating B210: " << o.usrp_args << "\n";
        auto usrp = uhd::usrp::multi_usrp::make(o.usrp_args);
        usrp->set_rx_rate(o.rate, o.channel);
        usrp->set_rx_freq(o.frequency, o.channel);
        usrp->set_rx_gain(o.gain, o.channel);
        usrp->set_rx_bandwidth(o.bandwidth, o.channel);
        usrp->set_rx_antenna(o.antenna, o.channel);
        std::cout << std::setprecision(17)
                  << "Actual rate: " << usrp->get_rx_rate(o.channel) << "\n"
                  << "Actual frequency: " << usrp->get_rx_freq(o.channel) << "\n"
                  << "Actual gain: " << usrp->get_rx_gain(o.channel) << "\n"
                  << "Actual RX bandwidth: " << usrp->get_rx_bandwidth(o.channel) << "\n"
                  << "Actual antenna: " << usrp->get_rx_antenna(o.channel) << "\n";

        uhd::stream_args_t stream_args("sc16", "sc16");
        stream_args.args = "num_recv_frames=4096";
        stream_args.channels = {o.channel};
        auto rx_stream = usrp->get_rx_stream(stream_args);
        const size_t block_samples = std::min(o.block_samples, rx_stream->get_max_num_samps());
        std::vector<std::complex<int16_t>> buffer(block_samples);
        uhd::rx_metadata_t metadata;
        std::ofstream iq(o.output, std::ios::binary);
        if (!iq) {
            throw std::runtime_error("cannot write IQ: " + o.output);
        }

        uhd::stream_cmd_t start(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
        start.stream_now = true;
        rx_stream->issue_stream_cmd(start);
        const size_t target_samples = static_cast<size_t>(o.duration * usrp->get_rx_rate(o.channel));
        size_t samples = 0;
        size_t overflows = 0;
        size_t timeouts = 0;
        size_t other_errors = 0;
        while (samples < target_samples) {
            const size_t requested = std::min(block_samples, target_samples - samples);
            const size_t received = rx_stream->recv(buffer.data(), requested, metadata, 2.0);
            if (metadata.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
                ++timeouts;
                continue;
            }
            if (metadata.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
                ++overflows;
            } else if (metadata.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
                ++other_errors;
            }
            if (received > 0) {
                iq.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(received * sizeof(buffer[0])));
                samples += received;
            }
        }
        uhd::stream_cmd_t stop(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS);
        rx_stream->issue_stream_cmd(stop);
        iq.close();
        const std::string status = (samples == target_samples && overflows == 0 && other_errors == 0)
            ? "passed" : "completed_with_errors";
        write_metadata(o, usrp, samples, overflows, timeouts, other_errors, status);
        std::cout << "Done. samples=" << samples << " target=" << target_samples
                  << " overflows=" << overflows << " timeouts=" << timeouts
                  << " other_errors=" << other_errors << "\n";
        return status == "passed" ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
