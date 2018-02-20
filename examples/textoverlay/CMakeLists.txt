buildExample(textoverlay textoverlay.cpp)

compile_shaders(textoverlay
    shaders/background.frag
    shaders/background.vert
    shaders/mesh.frag
    shaders/mesh.vert
    shaders/text.frag
    shaders/text.vert
)

link_assets(textoverlay
    models/cube.dae
    textures/skysphere_bc3_unorm.ktx
    textures/skysphere_astc_8x8_unorm.ktx
    textures/skysphere_etc2_unorm.ktx
    textures/round_window_astc_8x8_unorm.ktx
    textures/round_window_etc2_unorm.ktx
    textures/round_window_bc3_unorm.ktx
)
