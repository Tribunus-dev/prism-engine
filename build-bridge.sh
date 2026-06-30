#!/bin/bash
# build-bridge.sh — Build prism-bridge Rust crate and copy generated
# Swift bindings + dylib into the PrismAgent Xcode project.
#
# Run from the prism-engine repo root:
#   ./build-bridge.sh
#
# Then in Xcode:
#   1. Add the copied files to the project (drag into Xcode)
#   2. Add libprism_bridge.dylib to "Link Binary With Libraries"
#   3. Build & run

set -euo pipefail

cd "$(dirname "$0")"

PROJECT_ROOT="$PWD"
AGENT_DIR="$HOME/Developer/GitHub/PrismAgent/PrismAgent"
BRIDGE_DIR="$PROJECT_ROOT/swift-bindings"
TARGET_DIR="$PROJECT_ROOT/target/debug"

echo "→ Building prism-bridge..."
cargo build -p prism-bridge

echo "→ Generating Swift bindings..."
cargo run -p prism-bridge --bin uniffi-bindgen generate \
    --library "$TARGET_DIR/libprism_bridge.dylib" \
    --language swift \
    --out-dir "$BRIDGE_DIR"

echo "→ Copying to PrismAgent project..."
cp "$BRIDGE_DIR/prism_bridge.swift" "$AGENT_DIR/"
cp "$BRIDGE_DIR/prism_bridgeFFI.h" "$AGENT_DIR/"
cp "$BRIDGE_DIR/prism_bridgeFFI.modulemap" "$AGENT_DIR/"
cp "$TARGET_DIR/libprism_bridge.dylib" "$AGENT_DIR/"

echo "→ Done.  Open PrismAgent.xcodeproj and add these files:"
echo "   - prism_bridge.swift"
echo "   - prism_bridgeFFI.h"
echo "   - prism_bridgeFFI.modulemap"
echo "   - libprism_bridge.dylib"
