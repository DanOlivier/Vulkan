buildExample(texturecubemap texturecubemap.cpp)

copy_assets(texturecubemap
    textures/cubemap_yokohama_bc3_unorm.ktx
    textures/cubemap_yokohama_astc_8x8_unorm.ktx
    textures/cubemap_yokohama_etc2_unorm.ktx
    models/cube.obj
    models/sphere.obj
    models/teapot.dae
    models/torusknot.obj
)

