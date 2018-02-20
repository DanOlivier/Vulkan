file(GLOB BASE_SRC "*.cpp" 
    ../external/imgui/imgui.cpp 
    ../external/imgui/imgui_draw.cpp
    ../external/imgui/imgui_demo.cpp
)
file(GLOB BASE_HEADERS "*.hpp")

if(WIN32)
    add_library(base STATIC ${BASE_SRC})
    target_link_libraries(base ${Vulkan_LIBRARY} ${ASSIMP_LIBRARIES} ${WINLIBS})
 else(WIN32)
    add_library(base STATIC ${BASE_SRC})
    target_link_libraries(base ${Vulkan_LIBRARY} ${ASSIMP_LIBRARIES} ${XCB_LIBRARIES} ${WAYLAND_CLIENT_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif(WIN32)

 
list(APPEND SHADERS shaders/textoverlay.vert)
list(APPEND SHADERS shaders/textoverlay.frag)
list(APPEND SHADERS shaders/uioverlay.vert)
list(APPEND SHADERS shaders/uioverlay.frag)

set_target_properties(base PROPERTIES SHADERS "${SHADERS}")
