buildExample(offscreen offscreen.cpp)

compile_shaders(offscreen
    shaders/mirror.frag
    shaders/mirror.vert
    shaders/phong.frag
    shaders/phong.vert
    shaders/quad.frag
    shaders/quad.vert
)

link_assets(offscreen
    models/plane.obj
    models/chinesedragon.dae
    textures/darkmetal_bc3_unorm.ktx
    textures/darkmetal_astc_8x8_unorm.ktx
    textures/darkmetal_etc2_unorm.ktx
)

