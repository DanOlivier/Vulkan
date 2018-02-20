buildExample(imgui
    main.cpp
SHADERS
    shaders/scene.frag
    shaders/scene.vert
    shaders/ui.frag
    shaders/ui.vert
ASSETS
    models/vulkanscenemodels.dae
    models/vulkanscenebackground.dae
    models/vulkanscenelogos.dae
)

target_compile_options(imgui PRIVATE
    -fpermissive
    -Wno-overlength-strings
)