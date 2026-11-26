# Portal-ML Source Engine C++ Server Plugin

32-bit Win32 C++17 server plugin (`ISERVERPLUGINCALLBACKS003`) exposing the 3-layer Blackboard / Motor Primitive bridge for **Portal 1** (`appID=400`, Source 2013 SP) over a low-latency TCP IPC socket (`127.0.0.1:27500`, 4-byte length-prefixed UTF-8 JSON matching `portal_ml/bridge/protocol.py`).

## Verified Interface Match (Portal 1)
- Exported via `CreateInterface`: `ISERVERPLUGINCALLBACKS003` (zero `tier0.lib`/`tier1.lib` link dependencies)
- Consumed from `engine.dll`: `VEngineServer021`
- Consumed from `server.dll`: `PlayerInfoManager002`, `ServerGameDLL009`

## Building (MSVC 32-bit x86)

```powershell
cmake -S source_plugin -B source_plugin/build-x86 -G "Visual Studio 16 2019" -A Win32 -DPORTAL_ML_USE_SOURCE_SDK=ON
cmake --build source_plugin/build-x86 --config Release
```

## Installing & Running in Portal 1

1. Generate and compile `portal_ml_5chambers.bsp`:
   ```powershell
   python scripts/generate_vmf.py
   maps\compile_map.bat
   ```
2. Install the plugin into `<Portal>\portal\addons\`:
   - `portal_ml_plugin.dll`
   - `portal_ml_plugin.vdf` (`"Plugin" { "file" "addons/portal_ml_plugin" }`)
3. Launch Portal 1 with the map and fixed-timestep lock:
   ```powershell
   .\scripts\launch_portal.ps1
   ```
4. Verify live IPC connection and in-place chamber resets:
   ```powershell
   python scripts/test_live_portal.py
   ```
5. Run evolution directly against Portal 1:
   ```powershell
   python scripts/train_mock.py --backend source --host 127.0.0.1 --port 27500
   ```
