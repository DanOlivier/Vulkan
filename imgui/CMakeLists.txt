file(GLOB IMGUI_SOURCE ../external/imgui/*.cpp)

buildExample(imgui main.cpp ${IMGUI_SOURCE})

compile_shaders(imgui
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/scene.frag
    shaders/scene.vert
    shaders/ui.frag
    shaders/ui.vert
)

copy_assets(imgui
    models/vulkanscenemodels.dae
    models/vulkanscenebackground.dae
    models/vulkanscenelogos.dae
)