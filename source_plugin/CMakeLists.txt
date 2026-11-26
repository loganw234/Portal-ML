cmake_minimum_required(VERSION 3.16)
project(portal_ml_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PORTAL_ML_USE_SOURCE_SDK "Build against Source SDK 2013 public headers" ON)
set(SOURCE_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/source-sdk-2013/src" CACHE PATH "Path to source-sdk-2013/src")

set(PLUGIN_SOURCES
    src/ipc_server.cpp
    src/blackboard_extractor.cpp
    src/motor_controller.cpp
    src/portal_ml_plugin.cpp
)

add_library(portal_ml_plugin SHARED ${PLUGIN_SOURCES})
target_include_directories(portal_ml_plugin PRIVATE src)

if(WIN32)
    target_link_libraries(portal_ml_plugin PRIVATE ws2_32)
    target_compile_definitions(portal_ml_plugin PRIVATE
        _WIN32_WINNT=0x0601
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
        WIN32
    )
endif()

if(PORTAL_ML_USE_SOURCE_SDK)
    if(NOT EXISTS "${SOURCE_SDK_DIR}/public/engine/iserverplugin.h")
        message(FATAL_ERROR "Source SDK headers not found at ${SOURCE_SDK_DIR}/public/engine/iserverplugin.h")
    endif()
    target_compile_definitions(portal_ml_plugin PRIVATE
        PORTAL_ML_USE_SOURCE_SDK=1
        VALVE_LITTLE_ENDIAN=1
        COMPILER_MSVC=1
        COMPILER_MSVC32=1
    )
    target_include_directories(portal_ml_plugin PRIVATE
        "${SOURCE_SDK_DIR}/public"
        "${SOURCE_SDK_DIR}/public/tier0"
        "${SOURCE_SDK_DIR}/public/tier1"
        "${SOURCE_SDK_DIR}/public/engine"
        "${SOURCE_SDK_DIR}/public/mathlib"
        "${SOURCE_SDK_DIR}/public/game/server"
    )
endif()
