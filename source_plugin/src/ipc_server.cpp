#include "ipc_server.h"

#include <cstring>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace portal_ml {

namespace {
constexpr uint32_t kMaxFrameBytes = 4u * 1024u * 1024u;

void CloseSocketFd(intptr_t& fd) {
    if (fd < 0) return;
#ifdef _WIN32
    closesocket(static_cast<SOCKET>(fd));
#else
    close(static_cast<int>(fd));
#endif
    fd = -1;
}

void SetNonBlocking(intptr_t fd, bool non_blocking) {
#ifdef _WIN32
    u_long mode = non_blocking ? 1 : 0;
    ioctlsocket(static_cast<SOCKET>(fd), FIONBIO, &mode);
#else
    int flags = fcntl(static_cast<int>(fd), F_GETFL, 0);
    if (flags >= 0) {
        fcntl(static_cast<int>(fd), F_SETFL, non_blocking ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK));
    }
#endif
}
}  // namespace

IpcServer::~IpcServer() {
    Stop();
}

bool IpcServer::StartTcp(const std::string& host, uint16_t port) {
#ifdef _WIN32
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        return false;
    }
#endif
    initialized_ = true;

    auto s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#ifdef _WIN32
    if (s == INVALID_SOCKET) return false;
#else
    if (s < 0) return false;
#endif
    listen_fd_ = static_cast<intptr_t>(s);

    int opt = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        Stop();
        return false;
    }
    if (listen(s, 4) != 0) {
        Stop();
        return false;
    }
    SetNonBlocking(listen_fd_, true);
    return true;
}

void IpcServer::Stop() {
    DisconnectClient();
    CloseSocketFd(listen_fd_);
#ifdef _WIN32
    if (initialized_) {
        WSACleanup();
    }
#endif
    initialized_ = false;
}

bool IpcServer::PollAccept() {
    if (listen_fd_ < 0 || client_fd_ >= 0) {
        return client_fd_ >= 0;
    }
    sockaddr_in peer{};
#ifdef _WIN32
    int len = sizeof(peer);
    SOCKET c = accept(static_cast<SOCKET>(listen_fd_), reinterpret_cast<sockaddr*>(&peer), &len);
    if (c == INVALID_SOCKET) return false;
#else
    socklen_t len = sizeof(peer);
    int c = accept(static_cast<int>(listen_fd_), reinterpret_cast<sockaddr*>(&peer), &len);
    if (c < 0) return false;
#endif
    int nodelay = 1;
    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));
    client_fd_ = static_cast<intptr_t>(c);
    SetNonBlocking(client_fd_, false);
    return true;
}

bool IpcServer::HasClient() const {
    return client_fd_ >= 0;
}

void IpcServer::DisconnectClient() {
    CloseSocketFd(client_fd_);
}

bool IpcServer::RecvExact(void* buf, size_t n) {
    if (client_fd_ < 0) return false;
    auto* ptr = static_cast<char*>(buf);
    size_t remaining = n;
    while (remaining > 0) {
#ifdef _WIN32
        int got = recv(static_cast<SOCKET>(client_fd_), ptr, static_cast<int>(remaining), 0);
#else
        ssize_t got = recv(static_cast<int>(client_fd_), ptr, remaining, 0);
#endif
        if (got <= 0) {
            DisconnectClient();
            return false;
        }
        ptr += got;
        remaining -= static_cast<size_t>(got);
    }
    return true;
}

bool IpcServer::SendExact(const void* buf, size_t n) {
    if (client_fd_ < 0) return false;
    const auto* ptr = static_cast<const char*>(buf);
    size_t remaining = n;
    while (remaining > 0) {
#ifdef _WIN32
        int sent = send(static_cast<SOCKET>(client_fd_), ptr, static_cast<int>(remaining), 0);
#else
        ssize_t sent = send(static_cast<int>(client_fd_), ptr, remaining, 0);
#endif
        if (sent <= 0) {
            DisconnectClient();
            return false;
        }
        ptr += sent;
        remaining -= static_cast<size_t>(sent);
    }
    return true;
}

bool IpcServer::RecvFrame(std::string& out_json) {
    uint8_t hdr[4]{};
    if (!RecvExact(hdr, 4)) return false;
    const uint32_t len =
        static_cast<uint32_t>(hdr[0]) |
        (static_cast<uint32_t>(hdr[1]) << 8) |
        (static_cast<uint32_t>(hdr[2]) << 16) |
        (static_cast<uint32_t>(hdr[3]) << 24);
    if (len == 0 || len > kMaxFrameBytes) {
        DisconnectClient();
        return false;
    }
    out_json.resize(len);
    return RecvExact(out_json.data(), len);
}

bool IpcServer::SendFrame(const std::string& json) {
    if (json.size() > kMaxFrameBytes) return false;
    const auto len = static_cast<uint32_t>(json.size());
    uint8_t hdr[4] = {
        static_cast<uint8_t>(len & 0xFFu),
        static_cast<uint8_t>((len >> 8) & 0xFFu),
        static_cast<uint8_t>((len >> 16) & 0xFFu),
        static_cast<uint8_t>((len >> 24) & 0xFFu),
    };
    return SendExact(hdr, 4) && SendExact(json.data(), json.size());
}

}  // namespace portal_ml
