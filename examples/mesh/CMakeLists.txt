buildExample(mesh mesh.cpp)

compile_shaders(mesh
    shaders/mesh.frag
    shaders/mesh.vert
)

link_assets(mesh
    models/voyager/voyager.dae
    models/voyager/voyager_bc3_unorm.ktx
    models/voyager/voyager_astc_8x8_unorm.ktx
    models/voyager/voyager_etc2_unorm.ktx
)
