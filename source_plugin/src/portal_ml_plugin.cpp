#include "portal_ml_plugin.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

#ifdef PORTAL_ML_USE_SOURCE_SDK
#include "eiface.h"
#include "engine/iserverplugin.h"
#include "game/server/iplayerinfo.h"
#endif

namespace portal_ml {

namespace {
int ExtractIntField(const std::string& json, const std::string& key, int fallback) {
    const std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return fallback;
    pos += needle.size();
    return std::atoi(json.c_str() + pos);
}
}  // namespace

void PortalMLPlugin::SendServerCmd(const std::string& cmd, bool execute_now) const {
#ifdef PORTAL_ML_USE_SOURCE_SDK
    auto* engine = static_cast<IVEngineServer*>(engine_);
    if (engine) {
        engine->ServerCommand(cmd.c_str());
        if (execute_now) {
            engine->ServerExecute();
        }
    }
#else
    (void)cmd;
    (void)execute_now;
#endif
}

bool PortalMLPlugin::Load(void* engine_server, void* player_info_mgr, void* server_game_dll) {
    engine_ = engine_server;
    extractor_.SetEngineInterfaces(engine_server, player_info_mgr, server_game_dll);
    motor_.SetEngineServer(engine_server);
    return ipc_.StartTcp("127.0.0.1", 27500);
}

void PortalMLPlugin::Unload() {
    motor_.StopMovement();
    ipc_.Stop();
}

void PortalMLPlugin::ServerActivate() {
    SendServerCmd("sv_cheats 1; host_framerate 0.05; sv_gravity 600\n", false);
    ResetChamberInPlace(1);
}

void PortalMLPlugin::ResetChamberInPlace(int chamber_id) {
    current_chamber_ = chamber_id;
    tick_ = 0;
    pending_subticks_ = 0;
    awaiting_post_step_ = false;
    extractor_.OnResetChamber(chamber_id);

    const auto& layout = GetChamberLayout(chamber_id);
    std::ostringstream ss;
    // Sub-millisecond in-place reset (no changelevel):
    // 1. Stop movement and fizzle portals
    ss << "-forward; -use; ent_fire prop_portal Fizzle; "
       << "ent_fire portal_ml_blue SetActivatedState 0; "
       << "ent_fire portal_ml_orange SetActivatedState 0; ";
    // 2. Teleport player to chamber spawn
    ss << "setpos " << layout.player_spawn.x << ' ' << layout.player_spawn.y << ' '
       << layout.player_spawn.z << "; setang 0 " << layout.player_yaw << " 0; ";
    // 3. Reset chamber cube if present
    if (layout.has_cube) {
        ss << "ent_fire " << layout.cube_id << " AddOutput \"origin "
           << layout.cube_spawn.x << ' ' << layout.cube_spawn.y << ' '
           << layout.cube_spawn.z << "\"; ";
    }
    // 4. Reset exit door to initial state
    ss << "ent_fire " << layout.exit_id << (layout.exit_starts_open ? " Open" : " Close") << "\n";
    SendServerCmd(ss.str(), false);
}

void PortalMLPlugin::GameFrame(bool simulating) {
    if (!simulating) return;

    ipc_.PollAccept();
    if (!ipc_.HasClient()) return;

    // If a step command is executing across sub-ticks, wait until physics advances
    if (pending_subticks_ > 0) {
        --pending_subticks_;
        if (pending_subticks_ > 0) {
            return;
        }
        if (awaiting_post_step_) {
            awaiting_post_step_ = false;
            const auto post_state = extractor_.Extract(current_chamber_, tick_, fixed_dt_ * 2.0f);
            ipc_.SendFrame(
                "{\"ok\":true,\"op\":\"step\",\"snapshot\":" +
                BlackboardExtractor::ToJson(post_state) +
                ",\"reward\":" + (post_state.success ? "99.95" : "-0.05") +
                ",\"done\":" + (post_state.done ? "true" : "false") +
                ",\"success\":" + (post_state.success ? "true" : "false") +
                ",\"milestones\":[]}");
            return;
        }
    }

    std::string req;
    if (!ipc_.RecvFrame(req)) {
        return;
    }

    if (req.find("\"op\":\"ping\"") != std::string::npos) {
        ipc_.SendFrame("{\"ok\":true,\"op\":\"pong\"}");
        return;
    }

    if (req.find("\"op\":\"reset\"") != std::string::npos) {
        const int cid = ExtractIntField(req, "chamber_id", 1);
        ResetChamberInPlace(cid);
        const auto state = extractor_.Extract(current_chamber_, tick_, fixed_dt_ * 2.0f);
        ipc_.SendFrame("{\"ok\":true,\"op\":\"reset\",\"snapshot\":" +
                       BlackboardExtractor::ToJson(state) + "}");
        return;
    }

    if (req.find("\"op\":\"step\"") != std::string::npos) {
        ++tick_;
        const auto pre_state = extractor_.Extract(current_chamber_, tick_, fixed_dt_ * 2.0f);
        const auto cmd = MotorController::ParseCommandJson(req);
        motor_.Execute(cmd, pre_state, extractor_);
        // Advance 2 fixed physics ticks (0.10s) before returning the post-step snapshot
        pending_subticks_ = 2;
        awaiting_post_step_ = true;
        return;
    }

    if (req.find("\"op\":\"shutdown\"") != std::string::npos) {
        ipc_.SendFrame("{\"ok\":true,\"op\":\"shutdown\"}");
        ipc_.DisconnectClient();
        return;
    }

    ipc_.SendFrame("{\"ok\":false,\"error\":\"Unsupported operation\"}");
}

}  // namespace portal_ml

#ifdef PORTAL_ML_USE_SOURCE_SDK

// Official Valve IServerPluginCallbacks (ISERVERPLUGINCALLBACKS003) adapter.
// Exported directly via CreateInterface with zero tier0/tier1 .lib dependencies.
class CPortalMLServerPlugin final : public IServerPluginCallbacks {
public:
    bool Load(CreateInterfaceFn interfaceFactory, CreateInterfaceFn gameServerFactory) override {
        void* engine = interfaceFactory ? interfaceFactory(INTERFACEVERSION_VENGINESERVER_VERSION_21, nullptr) : nullptr;
        void* pim = gameServerFactory ? gameServerFactory(INTERFACEVERSION_PLAYERINFOMANAGER, nullptr) : nullptr;
        void* gamedll = gameServerFactory ? gameServerFactory(INTERFACEVERSION_SERVERGAMEDLL_VERSION_9, nullptr) : nullptr;
        const bool ok = impl_.Load(engine, pim, gamedll);
        if (engine) {
            static_cast<IVEngineServer*>(engine)->LogPrint(
                ok ? "[Portal-ML] Plugin loaded; IPC listening on 127.0.0.1:27500\n"
                   : "[Portal-ML] ERROR: Failed to bind IPC on 127.0.0.1:27500\n");
        }
        return ok;
    }

    void Unload() override { impl_.Unload(); }
    void Pause() override {}
    void UnPause() override {}
    const char* GetPluginDescription() override {
        return "Portal-ML Decoupled Neuroevolutionary Agent Bridge v0.1";
    }
    void LevelInit(char const* /*pMapName*/) override {}
    void ServerActivate(edict_t* /*pEdictList*/, int /*edictCount*/, int /*clientMax*/) override {
        impl_.ServerActivate();
    }
    void GameFrame(bool simulating) override { impl_.GameFrame(simulating); }
    void LevelShutdown() override {}
    void ClientActive(edict_t* /*pEntity*/) override {}
    void ClientDisconnect(edict_t* /*pEntity*/) override {}
    void ClientPutInServer(edict_t* /*pEntity*/, char const* /*playername*/) override {}
    void SetCommandClient(int /*index*/) override {}
    void ClientSettingsChanged(edict_t* /*pEdict*/) override {}
    PLUGIN_RESULT ClientConnect(bool* bAllowConnect, edict_t* /*pEntity*/, const char* /*pszName*/,
                                const char* /*pszAddress*/, char* /*reject*/, int /*maxrejectlen*/) override {
        if (bAllowConnect) *bAllowConnect = true;
        return PLUGIN_CONTINUE;
    }
    PLUGIN_RESULT ClientCommand(edict_t* /*pEntity*/, const CCommand& /*args*/) override {
        return PLUGIN_CONTINUE;
    }
    PLUGIN_RESULT NetworkIDValidated(const char* /*pszUserName*/, const char* /*pszNetworkID*/) override {
        return PLUGIN_CONTINUE;
    }
    void OnQueryCvarValueFinished(QueryCvarCookie_t /*iCookie*/, edict_t* /*pPlayerEntity*/,
                                  EQueryCvarValueStatus /*eStatus*/, const char* /*pCvarName*/,
                                  const char* /*pCvarValue*/) override {}
    void OnEdictAllocated(edict_t* /*edict*/) override {}
    void OnEdictFreed(const edict_t* /*edict*/) override {}

private:
    portal_ml::PortalMLPlugin impl_;
};

static CPortalMLServerPlugin g_PortalMLPlugin;

extern "C" __declspec(dllexport) void* CreateInterface(const char* pName, int* pReturnCode) {
    if (pName &&
        (std::strcmp(pName, INTERFACEVERSION_ISERVERPLUGINCALLBACKS) == 0 ||
         std::strcmp(pName, INTERFACEVERSION_ISERVERPLUGINCALLBACKS_VERSION_2) == 0 ||
         std::strcmp(pName, INTERFACEVERSION_ISERVERPLUGINCALLBACKS_VERSION_1) == 0)) {
        if (pReturnCode) *pReturnCode = IFACE_OK;
        return static_cast<IServerPluginCallbacks*>(&g_PortalMLPlugin);
    }
    if (pReturnCode) *pReturnCode = IFACE_FAILED;
    return nullptr;
}

#endif
