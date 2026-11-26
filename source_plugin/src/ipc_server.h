#pragma once

#include <cstdint>
#include <string>

namespace portal_ml {

// Low-latency length-prefixed JSON IPC server (matches portal_ml/bridge/protocol.py).
// Wire format: [4-byte little-endian uint32 length][UTF-8 JSON bytes]
class IpcServer {
public:
    IpcServer() = default;
    ~IpcServer();

    IpcServer(const IpcServer&) = delete;
    IpcServer& operator=(const IpcServer&) = delete;

    // Bind TCP localhost listener (127.0.0.1:port).
    bool StartTcp(const std::string& host, uint16_t port);
    void Stop();

    // Non-blocking accept check called from GameFrame().
    bool PollAccept();
    bool HasClient() const;

    // Blocking request/response exchange for locked fixed-timestep stepping.
    bool RecvFrame(std::string& out_json);
    bool SendFrame(const std::string& json);
    void DisconnectClient();

private:
    bool RecvExact(void* buf, size_t n);
    bool SendExact(const void* buf, size_t n);

    intptr_t listen_fd_ = -1;
    intptr_t client_fd_ = -1;
    bool initialized_ = false;
};

}  // namespace portal_ml
