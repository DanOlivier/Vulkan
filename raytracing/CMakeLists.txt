buildExample(raytracing raytracing.cpp)

compile_shaders(raytracing
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/texture.frag
    shaders/texture.vert
    shaders/raytracing.comp
)

copy_assets(raytracing
)

