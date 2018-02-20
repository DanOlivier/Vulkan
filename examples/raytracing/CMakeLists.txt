buildExample(raytracing raytracing.cpp)

compile_shaders(raytracing
    shaders/texture.frag
    shaders/texture.vert
    shaders/raytracing.comp
)

link_assets(raytracing
)

