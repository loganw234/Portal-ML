#pragma once

#include <string>

#include "blackboard_extractor.h"

namespace portal_ml {

enum class MotorCmdKind {
    kNoOp,
    kMoveTo,
    kMoveVec,
    kGrab,
    kDrop,
    kPress,
    kShootPortal,
};

struct MotorCommand {
    MotorCmdKind kind = MotorCmdKind::kNoOp;
    std::string entity_id;
    std::string color;    // "blue" | "orange"
    std::string wall_id;
    float forward = 0.0f;
    float strafe = 0.0f;
    float yaw_delta = 0.0f;
};

// Layer 3 (Phase 1): Deterministic scripted motor primitives executed inside Source.
// Phase 2 swaps individual Execute* methods for small neuroevolved neural networks.
class MotorController {
public:
    void SetEngineServer(void* engine_server) { engine_ = engine_server; }
    static MotorCommand ParseCommandJson(const std::string& json);
    void Execute(const MotorCommand& cmd, const BlackboardState& state, BlackboardExtractor& extractor) const;
    void StopMovement() const;

private:
    void IssueCmd(const std::string& cmd) const;
    void* engine_ = nullptr;
    mutable int stall_steps_ = 0;
    mutable int slide_remaining_ = 0;
    mutable float slide_sign_ = -1.0f;
    mutable float best_target_dist_ = 1e9f;
    mutable std::string last_target_id_;
    mutable int prev_zone_ = 0;
    mutable int pit_avoid_steps_ = 0;
    mutable float pit_avoid_sign_ = 1.0f;
    mutable std::string pit_avoid_target_;
};

}  // namespace portal_ml
