file(GLOB IMGUI_SOURCE ../external/imgui/*.cpp)

buildExample(imgui main.cpp ${IMGUI_SOURCE})

compile_shaders(imgui
    shaders/scene.frag
    shaders/scene.vert
    shaders/ui.frag
    shaders/ui.vert
)

