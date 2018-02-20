buildExample(radialblur radialblur.cpp)

compile_shaders(radialblur
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/colorpass.frag
    shaders/colorpass.vert
    shaders/phongpass.frag
    shaders/phongpass.vert
    shaders/radialblur.frag
    shaders/radialblur.vert
)

link_assets(radialblur
    models/glowsphere.dae
    textures/particle_gradient_rgba.ktx
)

