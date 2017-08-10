buildExample(texture texture.cpp)

compile_shaders(texture
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/texture.frag
    shaders/texture.vert
)

copy_assets(texture
    textures/metalplate01_rgba.ktx
)
