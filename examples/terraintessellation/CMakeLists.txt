buildExample(terraintessellation terraintessellation.cpp)

compile_shaders(terraintessellation
    shaders/skysphere.frag
    shaders/skysphere.vert
    shaders/terrain.frag
    shaders/terrain.tesc
    shaders/terrain.tese
    shaders/terrain.vert
)

link_assets(terraintessellation
    models/geosphere.obj
    textures/skysphere_bc3_unorm.ktx
    textures/skysphere_astc_8x8_unorm.ktx
    textures/skysphere_etc2_unorm.ktx
    textures/terrain_texturearray_astc_8x8_unorm.ktx
    textures/terrain_texturearray_bc3_unorm.ktx
    textures/terrain_texturearray_etc2_unorm.ktx
    textures/terrain_heightmap_r16.ktx
    textures/terrain_heightmap_r16.ktx
    textures/terrain_heightmap_r16.ktx
)

