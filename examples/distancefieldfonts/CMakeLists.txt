buildExample(distancefieldfonts distancefieldfonts.cpp)

compile_shaders(distancefieldfonts
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/bitmap.frag
    shaders/bitmap.vert
    shaders/sdf.frag
    shaders/sdf.vert
)

copy_assets(distancefieldfonts
    #font.fnt
    textures/font_sdf_rgba.ktx
    textures/font_bitmap_rgba.ktx
)

