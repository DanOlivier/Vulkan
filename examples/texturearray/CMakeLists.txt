buildExample(texturearray texturearray.cpp)

compile_shaders(texturearray
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/instancing.frag
    shaders/instancing.vert
)

link_assets(texturearray
    textures/texturearray_astc_8x8_unorm.ktx
    textures/texturearray_bc3_unorm.ktx
    textures/texturearray_etc2_unorm.ktx
)

