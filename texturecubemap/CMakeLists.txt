buildExample(texturecubemap texturecubemap.cpp)

compile_shaders(texturecubemap
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/cubemapdisplay.vert
    shaders/cubemapdisplay.frag

    shaders/skybox.vert
    shaders/skybox.frag

    shaders/reflect.vert
    shaders/reflect.frag
)

copy_assets(texturecubemap
    textures/cubemap_yokohama_bc3_unorm.ktx
    textures/cubemap_yokohama_astc_8x8_unorm.ktx
    textures/cubemap_yokohama_etc2_unorm.ktx

    models/cube.obj
    models/sphere.obj
    models/teapot.dae
    models/torusknot.obj
)

