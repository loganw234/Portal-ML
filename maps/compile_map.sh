#!/usr/bin/env bash
set -euo pipefail

BIN_DIR="${BIN_DIR:-$HOME/.steam/steam/steamapps/common/Portal/bin}"
GAME_DIR="${GAME_DIR:-$HOME/.steam/steam/steamapps/common/Portal/portal}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_VMF="$SCRIPT_DIR/portal_ml_5chambers.vmf"
MAP_BSP="$SCRIPT_DIR/portal_ml_5chambers.bsp"

echo "[1/3] Running vbsp..."
"$BIN_DIR/vbsp" -game "$GAME_DIR" "$MAP_VMF"

echo "[2/3] Running vvis (fast)..."
"$BIN_DIR/vvis" -fast -game "$GAME_DIR" "$MAP_BSP"

echo "[3/3] Running vrad (fast)..."
"$BIN_DIR/vrad" -fast -game "$GAME_DIR" "$MAP_BSP"

cp -f "$MAP_BSP" "$GAME_DIR/maps/portal_ml_5chambers.bsp"
echo "Done!"
