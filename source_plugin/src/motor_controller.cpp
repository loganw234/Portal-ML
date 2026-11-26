#include "motor_controller.h"

#include <cmath>
#include <sstream>

#ifdef PORTAL_ML_USE_SOURCE_SDK
#include "eiface.h"
#endif

namespace portal_ml {

namespace {
constexpr float kRadToDeg = 57.2957795f;

std::string ExtractJsonStringField(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\":\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    auto end = json.find('"', pos);
    if (end == std::string::npos) return {};
    return json.substr(pos, end - pos);
}

bool FindTargetPos(const BlackboardState& state, const std::string& id, Vec3& out_pos) {
    if (id == "portal:blue" && state.blue_portal.active) {
        out_pos = {
            state.blue_portal.pos.x + state.blue_portal.normal.x * 4.0f,
            state.blue_portal.pos.y + state.blue_portal.normal.y * 4.0f,
            state.blue_portal.pos.z,
        };
        return true;
    }
    if (id == "portal:orange" && state.orange_portal.active) {
        out_pos = {
            state.orange_portal.pos.x + state.orange_portal.normal.x * 4.0f,
            state.orange_portal.pos.y + state.orange_portal.normal.y * 4.0f,
            state.orange_portal.pos.z,
        };
        return true;
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
    IssueCmd("-forward; -use\n");
}

MotorCommand MotorController::ParseCommandJson(const std::string& json) {
    MotorCommand cmd;
    const std::string kind = ExtractJsonStringField(json, "cmd");
    if (kind == "move_to") {
        cmd.kind = MotorCmdKind::kMoveTo;
        cmd.entity_id = ExtractJsonStringField(json, "entity_id");
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
        StopMovement();
        return;
    }

    if (cmd.kind == MotorCmdKind::kMoveTo) {
        Vec3 target{};
        if (!FindTargetPos(state, cmd.entity_id, target)) {
            StopMovement();
            return;
        }
        const float dx = target.x - state.player_pos.x;
        const float dy = target.y - state.player_pos.y;
        const float yaw = std::atan2(dy, dx) * kRadToDeg;
        std::ostringstream ss;
        ss << "setang 0 " << yaw << " 0; +forward; -use\n";
        IssueCmd(ss.str());
        return;
    }

    if (cmd.kind == MotorCmdKind::kGrab) {
        Vec3 target{};
        if (!FindTargetPos(state, cmd.entity_id, target)) {
            StopMovement();
            return;
        }
        const float dx = target.x - state.player_pos.x;
        const float dy = target.y - state.player_pos.y;
        const float dz = (target.z + 12.0f) - (state.player_pos.z + 64.0f);
        const float dist_xy = std::max(1.0f, std::hypot(dx, dy));
        const float yaw = std::atan2(dy, dx) * kRadToDeg;
        const float pitch = -std::atan2(dz, dist_xy) * kRadToDeg;
        std::ostringstream ss;
        ss << "-forward; setang " << pitch << ' ' << yaw << " 0; +use; wait 2; -use\n";
        IssueCmd(ss.str());
        extractor.NotifyHoldState(cmd.entity_id);
        return;
    }

    if (cmd.kind == MotorCmdKind::kDrop) {
        IssueCmd("-forward; +use; wait 2; -use\n");
        extractor.NotifyHoldState("");
        return;
    }

    if (cmd.kind == MotorCmdKind::kPress) {
        std::ostringstream ss;
        ss << "-forward; ent_fire " << cmd.entity_id << " Press\n";
        IssueCmd(ss.str());
        extractor.NotifyPedestalPressed(8.0f);
        return;
    }

    if (cmd.kind == MotorCmdKind::kShootPortal) {
        for (const auto& w : state.portal_walls) {
            if (w.id == cmd.wall_id) {
                const float yaw = std::atan2(w.normal.y, w.normal.x) * kRadToDeg;
                const float px = w.pos.x + w.normal.x * 8.0f;
                const float py = w.pos.y + w.normal.y * 8.0f;
                const float pz = w.pos.z;
                const std::string ent_name = (cmd.color == "blue") ? "portal_ml_blue" : "portal_ml_orange";
                std::ostringstream ss;
                ss << "-forward; ent_fire " << ent_name << " NewLocation \""
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
