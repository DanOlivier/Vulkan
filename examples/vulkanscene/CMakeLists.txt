buildExample(vulkanscene vulkanscene.cpp)

compile_shaders(vulkanscene
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/logo.frag
    shaders/logo.vert
    shaders/mesh.frag
    shaders/mesh.vert
    shaders/skybox.frag
    shaders/skybox.vert
)

copy_assets(vulkanscene
    models/cube.obj
    models/vulkanscenelogos.dae
    models/vulkanscenebackground.dae
    models/vulkanscenemodels.dae
    textures/cubemap_vulkan.ktx
)

