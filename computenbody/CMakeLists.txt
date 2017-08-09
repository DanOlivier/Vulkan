buildExample(computenbody computenbody.cpp)

compile_shaders(computenbody
    shaders/particle.frag
    shaders/particle.vert
)

copy_assets(computenbody
    textures/particle01_rgba.ktx
    textures/particle_gradient_rgba.ktx
)

