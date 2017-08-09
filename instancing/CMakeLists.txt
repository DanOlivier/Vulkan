buildExample(instancing instancing.cpp)

compile_shaders(instancing
    shaders/instancing.frag
    shaders/instancing.vert
    shaders/planet.frag
    shaders/planet.vert
    shaders/starfield.frag
    shaders/starfield.vert
)

copy_assets(instancing
    models/rock01.dae
    models/sphere.obj
    textures/texturearray_rocks_astc_8x8_unorm.ktx
    textures/texturearray_rocks_bc3_unorm.ktx
    textures/texturearray_rocks_etc2_unorm.ktx
    textures/lavaplanet_astc_8x8_unorm.ktx
    textures/lavaplanet_bc3_unorm.ktx
    textures/lavaplanet_etc2_unorm.ktx
)

