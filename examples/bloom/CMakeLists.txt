buildExample(bloom bloom.cpp)

link_assets(bloom
    models/retroufo.dae
    models/retroufo_glow.dae
    models/cube.obj
    textures/cubemap_space.ktx
)

compile_shaders(bloom
    shaders/colorpass.frag
    shaders/colorpass.vert
    shaders/gaussblur.frag
    shaders/gaussblur.vert
    shaders/phongpass.frag
    shaders/phongpass.vert
    shaders/skybox.frag
    shaders/skybox.vert
)
