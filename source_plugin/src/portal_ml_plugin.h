#pragma once

#include "blackboard_extractor.h"
#include "ipc_server.h"
#include "motor_controller.h"

namespace portal_ml {

// Core Source Engine server plugin controller.
// Runs in locked fixed-timestep mode (host_framerate 0.05) and performs
// sub-millisecond in-place resets across all 5 chambers in portal_ml_5chambers.bsp.
class PortalMLPlugin {
public:
    bool Load(void* engine_server = nullptr, void* player_info_mgr = nullptr, void* server_game_dll = nullptr);
    void Unload();
    void ServerActivate();
    void GameFrame(bool simulating);

    // Sub-millisecond in-place reset (no changelevel):
    // teleports player & cubes, fizzles active portals, resets buttons/doors.
    void ResetChamberInPlace(int chamber_id);

private:
    void SendServerCmd(const std::string& cmd, bool execute_now = false) const;

    IpcServer ipc_;
    BlackboardExtractor extractor_;
    MotorController motor_;
    void* engine_ = nullptr;
    int current_chamber_ = 1;
    int tick_ = 0;
    int pending_subticks_ = 0;
    bool awaiting_post_step_ = false;
    float fixed_dt_ = 0.05f;
};

}  // namespace portal_ml
