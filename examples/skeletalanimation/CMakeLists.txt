buildExample(skeletalanimation skeletalanimation.cpp)

compile_shaders(skeletalanimation
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/mesh.frag
    shaders/mesh.vert
    shaders/texture.frag
    shaders/texture.vert
)

copy_assets(skeletalanimation
    models/goblin.dae
    models/plane_z.obj
    textures/goblin_astc_8x8_unorm.ktx
    textures/goblin_etc2_unorm.ktx
    textures/goblin_bc3_unorm.ktx
    textures/trail_bc3_unorm.ktx
    textures/trail_etc2_unorm.ktx
    textures/trail_astc_8x8_unorm.ktx
)

