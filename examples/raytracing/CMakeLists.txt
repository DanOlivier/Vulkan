buildExample(raytracing raytracing.cpp)

compile_shaders(raytracing
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/texture.frag
    shaders/texture.vert
    shaders/raytracing.comp
)

link_assets(raytracing
)

