buildExample(imgui main.cpp)
target_compile_options(imgui PRIVATE
    -fpermissive
    -Wno-overlength-strings
)

compile_shaders(imgui
    shaders/scene.frag
    shaders/scene.vert
    shaders/ui.frag
    shaders/ui.vert
)

link_assets(imgui
    models/vulkanscenemodels.dae
    models/vulkanscenebackground.dae
    models/vulkanscenelogos.dae
)
