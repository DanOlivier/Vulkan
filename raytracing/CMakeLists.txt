buildExample(raytracing raytracing.cpp)

compile_shaders(raytracing
    shaders/texture.frag
    shaders/texture.vert
)

copy_assets(raytracing
)

