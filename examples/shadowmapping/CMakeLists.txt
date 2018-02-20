buildExample(shadowmapping shadowmapping.cpp)

compile_shaders(shadowmapping
    shaders/offscreen.frag
    shaders/offscreen.vert
    shaders/quad.frag
    shaders/quad.vert
    shaders/scene.frag
    shaders/scene.vert
)

link_assets(shadowmapping
    models/vulkanscene_shadow.dae
    models/samplescene.dae
)

