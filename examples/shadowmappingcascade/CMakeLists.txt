buildExample(shadowmappingcascade shadowmappingcascade.cpp)

compile_shaders(shadowmappingcascade
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/debugshadowmap.vert
    shaders/debugshadowmap.frag

    shaders/depthpass.vert
    shaders/depthpass.frag

    shaders/scene.vert
    shaders/scene.frag
)

link_assets(shadowmappingcascade
    textures/gridlines.ktx
    textures/oak_bark.ktx
    textures/oak_leafs.ktx

    models/terrain_simple.dae
    models/oak_trunk.dae
    models/oak_leafs.dae
)