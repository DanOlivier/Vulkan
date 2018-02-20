buildExample(parallaxmapping parallaxmapping.cpp)

compile_shaders(parallaxmapping
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/parallax.frag
    shaders/parallax.vert
)

copy_assets(parallaxmapping
    models/plane_z.obj
    textures/rocks_normal_height_rgba.dds
    textures/rocks_color_bc3_unorm.dds
    textures/rocks_color_astc_8x8_unorm.ktx
    textures/rocks_color_etc2_unorm.ktx
)

