buildExample(shadowmapping shadowmapping.cpp)

compile_shaders(shadowmapping
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/offscreen.frag
    shaders/offscreen.vert
    shaders/quad.frag
    shaders/quad.vert
    shaders/scene.frag
    shaders/scene.vert
)

copy_assets(shadowmapping
    models/vulkanscene_shadow.dae
    models/samplescene.dae
)

