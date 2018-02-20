buildExample(texture texture.cpp)

compile_shaders(texture
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/texture.frag
    shaders/texture.vert
)

link_assets(texture
    textures/metalplate01_rgba.ktx
)
