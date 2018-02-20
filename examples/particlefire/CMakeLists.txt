buildExample(particlefire particlefire.cpp)

compile_shaders(particlefire
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/normalmap.frag
    shaders/normalmap.vert
    shaders/particle.frag
    shaders/particle.vert
)

link_assets(particlefire
    textures/particle_smoke.ktx
    textures/particle_fire.ktx
    textures/fireplace_colormap_astc_8x8_unorm.ktx
    textures/fireplace_normalmap_astc_8x8_unorm.ktx
    textures/fireplace_colormap_etc2_unorm.ktx
    textures/fireplace_normalmap_etc2_unorm.ktx
    textures/fireplace_colormap_bc3_unorm.ktx
    textures/fireplace_normalmap_bc3_unorm.ktx
    models/fireplace.obj
)

