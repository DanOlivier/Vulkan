buildExample(displacement displacement.cpp)

compile_shaders(displacement
    shaders/base.frag
    shaders/base.vert
    shaders/displacement.tesc
    shaders/displacement.tese
)

link_assets(displacement
    models/plane.obj
    textures/stonefloor03_color_bc3_unorm.ktx
    textures/stonefloor03_color_astc_8x8_unorm.ktx
    textures/stonefloor03_color_etc2_unorm.ktx
)

