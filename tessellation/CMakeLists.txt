buildExample(tessellation tessellation.cpp)

compile_shaders(tessellation
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/base.frag
    shaders/base.vert
    shaders/passthrough.tesc
    shaders/passthrough.tese
    shaders/pntriangles.tesc
    shaders/pntriangles.tese
)

copy_assets(tessellation
    models/lowpoly/deer.dae
    textures/deer_bc3_unorm.ktx
    textures/deer_astc_8x8_unorm.ktx
    textures/deer_etc2_unorm.ktx
)

