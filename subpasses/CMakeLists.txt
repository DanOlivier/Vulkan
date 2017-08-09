buildExample(subpasses subpasses.cpp)

compile_shaders(subpasses
    shaders/composition.frag
    shaders/composition.vert
    shaders/gbuffer.frag
    shaders/gbuffer.vert
    shaders/transparent.frag
    shaders/transparent.vert
)

copy_assets(subpasses
    models/samplebuilding.dae
    models/samplebuilding_glass.dae
    textures/colored_glass_bc3_unorm.ktx
    textures/colored_glass_astc_8x8_unorm.ktx
    textures/colored_glass_etc2_unorm.ktx
)
