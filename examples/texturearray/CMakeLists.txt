buildExample(texturearray texturearray.cpp)

compile_shaders(texturearray
    shaders/instancing.frag
    shaders/instancing.vert
)

link_assets(texturearray
    textures/texturearray_astc_8x8_unorm.ktx
    textures/texturearray_bc3_unorm.ktx
    textures/texturearray_etc2_unorm.ktx
)

