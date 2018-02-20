buildExample(vulkanscene vulkanscene.cpp)

compile_shaders(vulkanscene
    shaders/logo.frag
    shaders/logo.vert
    shaders/mesh.frag
    shaders/mesh.vert
    shaders/skybox.frag
    shaders/skybox.vert
)

link_assets(vulkanscene
    models/cube.obj
    models/vulkanscenelogos.dae
    models/vulkanscenebackground.dae
    models/vulkanscenemodels.dae
    textures/cubemap_vulkan.ktx
)

