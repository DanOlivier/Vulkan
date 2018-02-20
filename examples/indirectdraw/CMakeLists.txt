buildExample(indirectdraw indirectdraw.cpp)

compile_shaders(indirectdraw
    shaders/ground.frag
    shaders/ground.vert
    shaders/indirectdraw.frag
    shaders/indirectdraw.vert
    shaders/skysphere.frag
    shaders/skysphere.vert
)

link_assets(indirectdraw
    models/plants.dae
    models/plane_circle.dae
    models/skysphere.dae
    
    textures/texturearray_plants_astc_8x8_unorm.ktx
    textures/texturearray_plants_bc3_unorm.ktx
    textures/texturearray_plants_etc2_unorm.ktx
    
    textures/ground_dry_astc_8x8_unorm.ktx
    textures/ground_dry_bc3_unorm.ktx
    textures/ground_dry_etc2_unorm.ktx
)

