buildExample(computeparticles computeparticles.cpp)

compile_shaders(computeparticles
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/particle.comp
    shaders/particle.frag
    shaders/particle.vert
)

copy_assets(computeparticles
    textures/particle01_rgba.ktx
    textures/particle_gradient_rgba.ktx
)

