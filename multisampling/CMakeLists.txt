buildExample(multisampling multisampling.cpp)

compile_shaders(multisampling
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/mesh.frag
    shaders/mesh.vert
)

copy_assets(multisampling
    models/voyager/voyager.dae
    models/voyager/voyager_bc3_unorm.ktx
    models/voyager/voyager_astc_8x8_unorm.ktx
    models/voyager/voyager_etc2_unorm.ktx
)
