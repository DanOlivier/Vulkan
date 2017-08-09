buildExample(radialblur radialblur.cpp)

compile_shaders(radialblur
    shaders/colorpass.frag
    shaders/colorpass.vert
    shaders/phongpass.frag
    shaders/phongpass.vert
    shaders/radialblur.frag
    shaders/radialblur.vert
)

copy_assets(radialblur
    models/glowsphere.dae
    textures/particle_gradient_rgba.ktx
)

