buildExample(distancefieldfonts distancefieldfonts.cpp)

compile_shaders(distancefieldfonts
    shaders/bitmap.frag
    shaders/bitmap.vert
    shaders/sdf.frag
    shaders/sdf.vert
)

link_assets(distancefieldfonts
    #font.fnt
    textures/font_sdf_rgba.ktx
    textures/font_bitmap_rgba.ktx
)

