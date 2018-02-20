buildExample(texturemipmapgen texturemipmapgen.cpp)

compile_shaders(texturemipmapgen
    shaders/texture.frag
    shaders/texture.vert
)

link_assets(texturemipmapgen
    models/tunnel_cylinder.dae
    textures/metalplate_nomips_rgba.ktx
)

