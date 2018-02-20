buildExample(computeparticles computeparticles.cpp)

compile_shaders(computeparticles
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/particle.comp
    shaders/particle.frag
    shaders/particle.vert
)

link_assets(computeparticles
    textures/particle01_rgba.ktx
    textures/particle_gradient_rgba.ktx
)

