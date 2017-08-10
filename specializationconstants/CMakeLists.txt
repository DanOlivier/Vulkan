buildExample(specializationconstants specializationconstants.cpp)

compile_shaders(specializationconstants
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/uber.frag
    shaders/uber.vert
)

copy_assets(specializationconstants
    models/color_teapot_spheres.dae
    textures/metalplate_nomips_rgba.ktx
)

