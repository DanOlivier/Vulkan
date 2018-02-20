buildExample(texture texture.cpp)

compile_shaders(texture
    shaders/texture.frag
    shaders/texture.vert
)

link_assets(texture
    textures/metalplate01_rgba.ktx
)
