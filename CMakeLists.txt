cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
cmake_policy(VERSION 3.7)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(NAME vulkanExamples)

project(${NAME})

include(CMakeParseArguments)

include_directories(external)
find_package(glm REQUIRED)
find_package(gli REQUIRED)
find_package(ASSIMP REQUIRED)
include_directories(external/imgui)
include_directories(base)

OPTION(USE_D2D_WSI "Build the project using Direct to Display swapchain" OFF)
OPTION(USE_WAYLAND_WSI "Build the project using Wayland swapchain" OFF)

set(RESOURCE_INSTALL_DIR "" CACHE PATH "Path to install resources to (leave empty for running uninstalled)")

# Use FindVulkan module added with CMAKE 3.7
message(STATUS "Using module to find Vulkan")
find_package(Vulkan REQUIRED)

IF(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")
ELSE()
    find_package(Threads REQUIRED)
IF(USE_D2D_WSI)
    MESSAGE("Using direct to display extension...")
    add_definitions(-D_DIRECT2DISPLAY)
ELSEIF(USE_WAYLAND_WSI)
    find_package(Wayland REQUIRED)
    if (NOT WAYLAND_FOUND)
        message(FATAL_ERROR "Wayland development package not found")
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WAYLAND_KHR")
    include_directories(${WAYLAND_INCLUDE_DIR})
ELSE()
    find_package(XCB REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_XCB_KHR")
ENDIF()
    # Todo : android?
ENDIF()

IF (NOT Vulkan_FOUND)
    message(FATAL_ERROR "Could not find Vulkan library!")
ELSE()
    message(STATUS ${Vulkan_LIBRARY})
ENDIF()

find_program(GLSLANG_VALIDATOR NAMES glslangValidator)
if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found!")
endif()

# Set preprocessor defines
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -D_USE_MATH_DEFINES")

# Clang specific stuff
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch-enum")
endif()

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
if(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wextra -Wundef -Wno-unused-parameter -Wno-missing-field-initializers -Wno-unused-function")
endif(CMAKE_COMPILER_IS_GNUCXX)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-std=c++11)

function(copy_assets EXAMPLE_NAME)
    foreach(ASSET ${ARGN})
        set(input_file ${CMAKE_SOURCE_DIR}/data/${ASSET})
        get_filename_component(DIR_NAME ${ASSET} DIRECTORY)
        set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${ASSET})
        # Must do this in two steps to propagate value
        set(all_assets ${all_assets} ${output_file})
        set(all_assets ${all_assets} PARENT_SCOPE)
        #message("Generating: ${input_file} -> ${output_file}")
        add_custom_command(
            OUTPUT ${output_file} POST_BUILD
            COMMAND mkdir -p ${DIR_NAME}
            COMMAND cp ${input_file} ${output_file}
            DEPENDS ${input_file}
            COMMENT "Copying asset ${ASSET} to ${output_file}"
        )
    endforeach()
    add_custom_target(copy_assets-${EXAMPLE_NAME} ALL DEPENDS ${all_assets})
    add_dependencies(${EXAMPLE_NAME} copy_assets-${EXAMPLE_NAME})
    #file(COPY ${assets} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/assets)
endfunction()

function(compile_shaders EXAMPLE_NAME)
    set(SHADERS ${ARGN})
    # Add shaders
    #file(GLOB SHADERS
    #	RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    #	"shaders/*.vert" 
    #	"shaders/*.frag"
    #	"shaders/*.geom"
    #	"shaders/*.tesc"
    #	"shaders/*.tese"
    #)
    #source_group("Shaders" FILES ${SHADERS})
    #message("EXAMPLE_NAME: ${EXAMPLE_NAME}, SHADERS: ${SHADERS}")

    foreach(SHADER ${SHADERS})
        
        get_filename_component(FILE_NAME ${SHADER} NAME)
        set(output_file ${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.spv)
        set(compiled_shaders ${compiled_shaders} ${output_file})
        set(compiled_shaders ${compiled_shaders} PARENT_SCOPE)
        add_custom_command(
            OUTPUT ${output_file}
            COMMAND mkdir -p shaders
            COMMAND ${GLSLANG_VALIDATOR} -w -V ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${output_file}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
            COMMENT "Compiling shader ${output_file}"
        )
    endforeach()
    add_custom_target(shader-${EXAMPLE_NAME} ALL DEPENDS ${compiled_shaders})
    add_dependencies(${EXAMPLE_NAME} shader-${EXAMPLE_NAME})
endfunction()

add_custom_target(run-all)

#file(GLOB SOURCE *.cpp )

# Function for building single example
function(buildExample EXAMPLE_NAME)
    list(REMOVE_AT ARGV 0)
    cmake_parse_arguments(ARGS "" "" "TEXTURES;MODELS" ${ARGN})
    #message("ARGN: ${ARGN}")
    #message("ARGV: ${ARGV}")
    #message("ARGS_TEXTURES: ${ARGS_TEXTURES}")
    #message("ARGS_MODELS: ${ARGS_MODELS}")
    #message("ARGS_UNPARSED_ARGUMENTS: ${ARGS_UNPARSED_ARGUMENTS}")
    
    if(WIN32)
        add_executable(${EXAMPLE_NAME} WIN32 
            ${ARGS_UNPARSED_ARGUMENTS} 
            #${ARGS_TEXTURES}
            #${ARGS_MODELS}
            #${compiled_shaders}
        )
        target_link_libraries(${EXAMPLE_NAME}
            ${Vulkan_LIBRARY}
            ${ASSIMP_LIBRARIES}
            ${WINLIBS})
    else()
        add_executable(${EXAMPLE_NAME} 
            ${ARGS_UNPARSED_ARGUMENTS}
            #${ARGS_TEXTURES}
            #${ARGS_MODELS}
            #${compiled_shaders}
        )
        target_link_libraries(${EXAMPLE_NAME}
            base
            ${Vulkan_LIBRARY} #VkLayer_api_dump
            ${ASSIMP_LIBRARIES}
            ${XCB_LIBRARIES}
            ${WAYLAND_CLIENT_LIBRARIES}
            ${CMAKE_THREAD_LIBS_INIT})
    endif()
    #compile_shaders(${EXAMPLE_NAME})

    add_custom_target(run-${EXAMPLE_NAME}
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${EXAMPLE_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${EXAMPLE_NAME}
    )
    add_dependencies(run-all run-${EXAMPLE_NAME})

endfunction()

#if(RESOURCE_INSTALL_DIR)
#    add_definitions(-DVK_EXAMPLE_DATA_DIR=\"${RESOURCE_INSTALL_DIR}/\")
#    install(DIRECTORY data/ DESTINATION ${RESOURCE_INSTALL_DIR}/)
#else()
#    add_definitions(-DVK_EXAMPLE_DATA_DIR=\"../../data/\")
#endif()

# Compiler specific stuff
IF(MSVC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
ENDIF(MSVC)

IF(WIN32)
    # Nothing here (yet)
ELSE(WIN32)
    link_libraries(
        ${XCB_LIBRARIES} 
        ${Vulkan_LIBRARY}
        ${ASSIMP_LIBRARIES}
        ${WAYLAND_CLIENT_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )
ENDIF(WIN32)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")

add_subdirectory(base)
add_subdirectory(examples)