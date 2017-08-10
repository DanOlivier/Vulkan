cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(NAME vulkanExamples)
project(${NAME})

include(CMakeParseArguments)

include_directories(external)
#include_directories(external/glm)
#include_directories(external/gli)
include_directories(external/assimp)
include_directories(external/imgui)
include_directories(base)

OPTION(USE_D2D_WSI "Build the project using Direct to Display swapchain" OFF)
OPTION(USE_WAYLAND_WSI "Build the project using Wayland swapchain" OFF)

# Use FindVulkan module added with CMAKE 3.7
if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
	message(STATUS "Using module to find Vulkan")
	find_package(Vulkan)
endif()

if(WIN32)
	IF (NOT Vulkan_FOUND)
		find_library(Vulkan_LIBRARY NAMES vulkan-1 vulkan PATHS ${CMAKE_SOURCE_DIR}/libs/vulkan)
		IF (Vulkan_LIBRARY)
			set(Vulkan_FOUND ON)
			message("Using bundled Vulkan library version")
		endif()
	endif()
	find_library(ASSIMP_LIBRARIES NAMES assimp libassimp.dll.a PATHS ${CMAKE_SOURCE_DIR}/libs/assimp)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WIN32_KHR")
else()
	IF (NOT Vulkan_FOUND)
		find_library(Vulkan_LIBRARY NAMES vulkan HINTS "$ENV{VULKAN_SDK}/lib" "${CMAKE_SOURCE_DIR}/libs/vulkan" REQUIRED)
		IF (Vulkan_LIBRARY)
			set(Vulkan_FOUND ON)
			message("Using bundled Vulkan library version")
		endif()
	endif()
	find_package(ASSIMP REQUIRED)
	find_package(Threads REQUIRED)
if(USE_D2D_WSI)
	message("Using direct to display extension...")
	add_definitions(-D_DIRECT2DISPLAY)
elseif(USE_WAYLAND_WSI)
	find_package(Wayland REQUIRED)
	if (NOT WAYLAND_FOUND)
		message(FATAL_ERROR "Wayland development package not found")
	endif ()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_WAYLAND_KHR")
	include_directories(${WAYLAND_INCLUDE_DIR})
else()
	find_package(XCB REQUIRED)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_USE_PLATFORM_XCB_KHR")
endif()
	# Todo : android?
endif()

if(NOT Vulkan_FOUND)
	message(FATAL_ERROR "Could not find Vulkan library!")
else()
	message(STATUS ${Vulkan_LIBRARY})
endif()

find_program(GLSLANG_VALIDATOR NAMES glslangValidator)

# Set preprocessor defines
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -D_USE_MATH_DEFINES")

# Clang specific stuff
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch-enum")
endif()

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wextra -Wundef -Wno-unused-parameter -Wno-missing-field-initializers")
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
        #message("Generating: ${ASSET} -> ${output_file}")
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
	#message("SHADERS: ${SHADERS}")

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
			VulkanBase
			${Vulkan_LIBRARY} VkLayer_api_dump
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

# Compiler specific stuff
if(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")

add_subdirectory(base)

add_subdirectory(bloom)
add_subdirectory(computecloth)
add_subdirectory(computecullandlod)
add_subdirectory(computenbody)
add_subdirectory(computeparticles)
add_subdirectory(computeshader)
add_subdirectory(debugmarker)
add_subdirectory(deferred)
add_subdirectory(deferredmultisampling)
add_subdirectory(deferredshadows)
add_subdirectory(displacement)
add_subdirectory(distancefieldfonts)
add_subdirectory(dynamicuniformbuffer)
add_subdirectory(gears)
add_subdirectory(geometryshader)
add_subdirectory(hdr)
add_subdirectory(imgui)
add_subdirectory(indirectdraw)	
add_subdirectory(instancing)
add_subdirectory(mesh)
add_subdirectory(multisampling)
add_subdirectory(multithreading)
add_subdirectory(occlusionquery)
add_subdirectory(offscreen)
add_subdirectory(parallaxmapping)
add_subdirectory(particlefire)
add_subdirectory(pbrbasic)
add_subdirectory(pbribl)
add_subdirectory(pbrtexture)
add_subdirectory(pipelines)
add_subdirectory(pushconstants)
add_subdirectory(radialblur)
add_subdirectory(raytracing)
add_subdirectory(scenerendering)
add_subdirectory(screenshot)
add_subdirectory(shadowmapping)
add_subdirectory(shadowmappingomni)
add_subdirectory(skeletalanimation)
add_subdirectory(specializationconstants)
add_subdirectory(sphericalenvmapping)
add_subdirectory(ssao)
add_subdirectory(subpasses)
add_subdirectory(terraintessellation)
add_subdirectory(tessellation)
add_subdirectory(textoverlay)
add_subdirectory(texture)
add_subdirectory(texture3d)
add_subdirectory(texturearray)
add_subdirectory(texturecubemap)
add_subdirectory(texturemipmapgen)
add_subdirectory(texturesparseresidency)
add_subdirectory(triangle)
add_subdirectory(viewportarray)
add_subdirectory(vulkanscene)
