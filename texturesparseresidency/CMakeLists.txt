buildExample(texturesparseresidency texturesparseresidency.cpp)

compile_shaders(texturesparseresidency
    shaders/sparseresidency.frag
    shaders/sparseresidency.vert
)

copy_assets(texturesparseresidency
    textures/ground_dry_bc3_unorm.ktx
    textures/terrain_heightmap_r16.ktx
    textures/terrain_heightmap_r16.ktx
)

