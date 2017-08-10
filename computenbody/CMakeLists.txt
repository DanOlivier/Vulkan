buildExample(computenbody computenbody.cpp)

compile_shaders(computenbody
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/particle.frag
    shaders/particle.vert
    shaders/particle_calculate.comp
    shaders/particle_integrate.comp
)

copy_assets(computenbody
    textures/particle01_rgba.ktx
    textures/particle_gradient_rgba.ktx
)

