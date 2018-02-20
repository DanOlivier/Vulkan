buildExample(computeshader computeshader.cpp)

compile_shaders(computeshader
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/edgedetect.comp
    shaders/emboss.comp
    shaders/sharpen.comp
    shaders/texture.frag
    shaders/texture.vert
)

link_assets(computeshader
    textures/het_kanonschot_rgba8.ktx
)

