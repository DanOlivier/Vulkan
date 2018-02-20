buildExample(texturemipmapgen texturemipmapgen.cpp)

compile_shaders(texturemipmapgen
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/texture.frag
    shaders/texture.vert
)

copy_assets(texturemipmapgen
    models/tunnel_cylinder.dae
    textures/metalplate_nomips_rgba.ktx
)

