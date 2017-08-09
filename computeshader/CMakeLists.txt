buildExample(computeshader computeshader.cpp)

compile_shaders(computeshader
    shaders/texture.frag
    shaders/texture.vert
)

copy_assets(computeshader
    textures/het_kanonschot_rgba8.ktx
)

