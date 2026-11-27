#include "motor_controller.h"

#include <cmath>
#include <cstdlib>
#include <sstream>

#ifdef PORTAL_ML_USE_SOURCE_SDK
#include "eiface.h"
#endif

namespace portal_ml {

namespace {
constexpr float kRadToDeg = 57.2957795f;
constexpr float kMoveSpeed = 240.0f;

float ClampF(float v, float lo, float hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

float WrapDeg(float deg) {
    float r = std::fmod(deg + 180.0f, 360.0f);
    if (r < 0.0f) r += 360.0f;
    return r - 180.0f;
}

float ExtractJsonFloatField(const std::string& json, const std::string& key, float fallback = 0.0f) {
    const std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos += needle.size();
    return static_cast<float>(std::atof(json.c_str() + pos));
}

std::string ExtractJsonStringField(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    auto end = json.find('"', pos);
    if (end == std::string::npos) return {};
    return json.substr(pos, end - pos);
}

bool FindPortalApproach(const BlackboardState& state, const PortalRecord& portal, Vec3& out_pos) {
    if (!portal.active) return false;
    const float rx = state.player_pos.x - portal.pos.x;
    const float ry = state.player_pos.y - portal.pos.y;
    // Distance along wall normal and laterally parallel to wall
    const float along_n = rx * portal.normal.x + ry * portal.normal.y;
    const float lateral = std::fabs(-rx * portal.normal.y + ry * portal.normal.x);
    if (lateral > 28.0f && along_n < 44.0f) {
        // Line up cleanly in front of portal before walking straight through
        out_pos = {
            portal.pos.x + portal.normal.x * 56.0f,
            portal.pos.y + portal.normal.y * 56.0f,
            portal.pos.z,
        };
    } else {
        out_pos = {portal.pos.x, portal.pos.y, portal.pos.z};
    }
    return true;
}

bool FindTargetPos(const BlackboardState& state, const std::string& id, Vec3& out_pos) {
    if (id == "portal:blue") {
        return FindPortalApproach(state, state.blue_portal, out_pos);
    }
    if (id == "portal:orange") {
        return FindPortalApproach(state, state.orange_portal, out_pos);
    }
    for (const auto& e : state.entities) {
        if (e.id == id) {
            out_pos = e.pos;
            return true;
        }
    }
    return false;
}
}  // namespace

void MotorController::IssueCmd(const std::string& cmd) const {
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    if (engine) {
        engine->ServerCommand(cmd.c_str());
    }
#else
    (void)cmd;
#endif
}

void MotorController::StopMovement() const {
    IssueCmd("-forward; -use; ent_fire !player AddOutput \"basevelocity 0 0 0\"\n");
}

MotorCommand MotorController::ParseCommandJson(const std::string& json) {
    MotorCommand cmd;
    const std::string kind = ExtractJsonStringField(json, "cmd");
    if (kind == "move_to") {
        cmd.kind = MotorCmdKind::kMoveTo;
        cmd.entity_id = ExtractJsonStringField(json, "entity_id");
    } else if (kind == "move_vec") {
        cmd.kind = MotorCmdKind::kMoveVec;
        cmd.forward = ExtractJsonFloatField(json, "forward", 0.0f);
        cmd.strafe = ExtractJsonFloatField(json, "strafe", 0.0f);
        cmd.yaw_delta = ExtractJsonFloatField(json, "yaw_delta", 0.0f);
    } else if (kind == "grab") {
        cmd.kind = MotorCmdKind::kGrab;
        cmd.entity_id = ExtractJsonStringField(json, "entity_id");
    } else if (kind == "drop") {
        cmd.kind = MotorCmdKind::kDrop;
    } else if (kind == "press") {
        cmd.kind = MotorCmdKind::kPress;
        cmd.entity_id = ExtractJsonStringField(json, "entity_id");
    } else if (kind == "shoot_portal") {
        cmd.kind = MotorCmdKind::kShootPortal;
        cmd.color = ExtractJsonStringField(json, "color");
        cmd.wall_id = ExtractJsonStringField(json, "wall_id");
    }
    return cmd;
}

void MotorController::Execute(
    const MotorCommand& cmd,
    const BlackboardState& state,
    BlackboardExtractor& extractor) const {
    if (cmd.kind == MotorCmdKind::kNoOp) {
        extractor.SetPlayerVelocityXY(0.0f, 0.0f);
        StopMovement();
        return;
    }

    if (cmd.kind == MotorCmdKind::kMoveVec) {
        // Phase 2 continuous body-frame control from neuroevolved MLP
        float fwd = ClampF(cmd.forward, -1.0f, 1.0f);
        float stf = ClampF(cmd.strafe, -1.0f, 1.0f);
        const float yd = ClampF(cmd.yaw_delta, -1.0f, 1.0f);
        const float mag = std::hypot(fwd, stf);
        if (mag > 1.0f) {
            fwd /= mag;
            stf /= mag;
        }
        const float new_yaw = WrapDeg(state.player_yaw + yd * 36.0f);
        extractor.SetPlayerEyeAngles(0.0f, new_yaw);

        const float rad = new_yaw / kRadToDeg;
        const float cos_y = std::cos(rad);
        const float sin_y = std::sin(rad);
        const float target_vx = (fwd * cos_y - stf * sin_y) * kMoveSpeed;
        const float target_vy = (fwd * sin_y + stf * cos_y) * kMoveSpeed;
        const float dvx = ClampF(target_vx - state.player_vel.x, -260.0f, 260.0f);
        const float dvy = ClampF(target_vy - state.player_vel.y, -260.0f, 260.0f);

        std::ostringstream ss;
        ss << "ent_fire !player AddOutput \"basevelocity " << dvx << ' ' << dvy << " 0\"; ";
        if (!state.held_entity_id.empty()) {
            const float cx = state.player_pos.x;
            const float cy = state.player_pos.y;
            const float cz = state.player_pos.z + 108.0f;
            ss << "ent_fire " << state.held_entity_id << " AddOutput \"origin "
               << cx << ' ' << cy << ' ' << cz << "\"; ";
        }
        ss << "\n";
        IssueCmd(ss.str());
        return;
    }

    if (cmd.kind == MotorCmdKind::kMoveTo) {
        Vec3 target{};
        if (!FindTargetPos(state, cmd.entity_id, target)) {
            extractor.SetPlayerVelocityXY(0.0f, 0.0f);
            StopMovement();
            return;
        }
        const float dx = target.x - state.player_pos.x;
        const float dy = target.y - state.player_pos.y;
        const float dist = std::hypot(dx, dy);
        if (dist < 1e-3f) {
            extractor.SetPlayerVelocityXY(0.0f, 0.0f);
            StopMovement();
            return;
        }
        const float nx = dx / dist;
        const float ny = dy / dist;
        const bool is_portal = (cmd.entity_id == "portal:blue" || cmd.entity_id == "portal:orange");
        const float speed = (!is_portal && dist < 72.0f) ? 140.0f : kMoveSpeed;
        const float target_vx = nx * speed;
        const float target_vy = ny * speed;
        // Closed-loop delta-velocity impulse via basevelocity (prevents runaway stacking)
        const float dvx = ClampF(target_vx - state.player_vel.x, -260.0f, 260.0f);
        const float dvy = ClampF(target_vy - state.player_vel.y, -260.0f, 260.0f);
        const float yaw = std::atan2(dy, dx) * kRadToDeg;
        extractor.SetPlayerEyeAngles(0.0f, yaw);

        std::ostringstream ss;
        ss << "ent_fire !player AddOutput \"basevelocity " << dvx << ' ' << dvy << " 0\"; ";
        // Carry held cube well above 72u player hull (non-solid while carried)
        if (!state.held_entity_id.empty()) {
            const float cx = state.player_pos.x;
            const float cy = state.player_pos.y;
            const float cz = state.player_pos.z + 108.0f;
            ss << "ent_fire " << state.held_entity_id << " AddOutput \"origin "
               << cx << ' ' << cy << ' ' << cz << "\"; ";
        }
        ss << "\n";
        IssueCmd(ss.str());
        return;
    }

    if (cmd.kind == MotorCmdKind::kGrab) {
        Vec3 target{};
        if (!FindTargetPos(state, cmd.entity_id, target)) {
            extractor.SetPlayerVelocityXY(0.0f, 0.0f);
            StopMovement();
            return;
        }
        const float dx = target.x - state.player_pos.x;
        const float dy = target.y - state.player_pos.y;
        const float yaw = std::atan2(dy, dx) * kRadToDeg;
        const float cx = state.player_pos.x;
        const float cy = state.player_pos.y;
        const float cz = state.player_pos.z + 108.0f;

        extractor.SetPlayerVelocityXY(0.0f, 0.0f);
        std::ostringstream ss;
        ss << "setang 0 " << yaw << " 0; "
           << "ent_fire !player AddOutput \"basevelocity 0 0 0\"; "
           << "ent_fire " << cmd.entity_id << " DisableMotion; "
           << "ent_fire " << cmd.entity_id << " AddOutput \"solid 0\"; "
           << "ent_fire " << cmd.entity_id << " AddOutput \"origin "
           << cx << ' ' << cy << ' ' << cz << "\"\n";
        IssueCmd(ss.str());
        extractor.NotifyHoldState(cmd.entity_id);
        return;
    }

    if (cmd.kind == MotorCmdKind::kDrop) {
        extractor.SetPlayerVelocityXY(0.0f, 0.0f);
        std::ostringstream ss;
        ss << "ent_fire !player AddOutput \"basevelocity 0 0 0\"; ";
        if (!state.held_entity_id.empty()) {
            // Aim downward ~62 deg (~34u ahead) so ent_teleport drops the VPhysics cube
            // cleanly in front of the player without overlapping the player hull.
            float drop_yaw = state.player_yaw;
            float drop_pitch = 62.0f;
            for (const auto& e : state.entities) {
                if (e.kind == "floor_button" && e.zone_id == state.player_zone_id) {
                    const float dx = e.pos.x - state.player_pos.x;
                    const float dy = e.pos.y - state.player_pos.y;
                    const float d = std::hypot(dx, dy);
                    if (d <= 52.0f) {
                        if (d > 4.0f) {
                            drop_yaw = std::atan2(dy, dx) * kRadToDeg;
                        }
                        drop_pitch = std::atan2(64.0f, std::max(18.0f, d)) * kRadToDeg;
                    }
                    break;
                }
            }
            extractor.SetPlayerEyeAngles(drop_pitch, drop_yaw);
            // Restore solid 6 while cube is still overhead (z+108) so ent_teleport uses
            // its real +-20.3u bounding box and places it flush on top of func_weight_button.
            ss << "ent_fire " << state.held_entity_id << " AddOutput \"solid 6\"";
        }
        ss << "\n";
        IssueCmd(ss.str());
        return;
    }

    if (cmd.kind == MotorCmdKind::kPress) {
        extractor.SetPlayerVelocityXY(0.0f, 0.0f);
        std::ostringstream ss;
        ss << "ent_fire " << cmd.entity_id << " Press\n";
        IssueCmd(ss.str());
        extractor.NotifyPedestalPressed(8.0f);
        return;
    }

    if (cmd.kind == MotorCmdKind::kShootPortal) {
        extractor.SetPlayerVelocityXY(0.0f, 0.0f);
        for (const auto& w : state.portal_walls) {
            if (w.id == cmd.wall_id) {
                const float yaw = std::atan2(w.normal.y, w.normal.x) * kRadToDeg;
                // Place flush on the front face of the 1u concrete panel
                const float px = w.pos.x + w.normal.x * 1.5f;
                const float py = w.pos.y + w.normal.y * 1.5f;
                const float pz = w.pos.z;
                const std::string ent_name = (cmd.color == "blue") ? "portal_ml_blue" : "portal_ml_orange";
                std::ostringstream ss;
                ss << "ent_fire " << ent_name << " NewLocation \""
                   << px << ' ' << py << ' ' << pz << " 0 " << yaw << " 0\"; "
                   << "ent_fire " << ent_name << " SetActivatedState 1\n";
                IssueCmd(ss.str());
                extractor.NotifyPortalPlaced(cmd.color, cmd.wall_id);
                return;
            }
        }
    }
}

}  // namespace portal_ml
