buildExample(bloom bloom.cpp)

copy_assets(bloom
    models/retroufo.dae
    models/retroufo_glow.dae
    models/cube.obj
    textures/cubemap_space.ktx
)

compile_shaders(bloom
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/colorpass.frag
    shaders/colorpass.vert
    shaders/gaussblur.frag
    shaders/gaussblur.vert
    shaders/phongpass.frag
    shaders/phongpass.vert
    shaders/skybox.frag
    shaders/skybox.vert
)