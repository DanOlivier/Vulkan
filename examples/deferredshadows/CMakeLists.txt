buildExample(deferredshadows deferredshadows.cpp)

compile_shaders(deferredshadows
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/debug.frag
    shaders/debug.vert
    shaders/deferred.frag
    shaders/deferred.vert
    shaders/mrt.frag
    shaders/mrt.vert
    shaders/shadow.frag
    shaders/shadow.geom
    shaders/shadow.vert
)

link_assets(deferredshadows
    models/openbox.dae

    models/armor/armor.dae
    models/armor/armor.dae
    models/armor/color_astc_8x8_unorm.ktx
    models/armor/color_bc3_unorm.ktx
    models/armor/color_etc2_unorm.ktx
    models/armor/normal_astc_8x8_unorm.ktx
    models/armor/normal_bc3_unorm.ktx
    models/armor/normal_etc2_unorm.ktx

    textures/stonefloor02_color_astc_8x8_unorm.ktx
    textures/stonefloor02_color_bc3_unorm.ktx
    textures/stonefloor02_color_etc2_unorm.ktx
    textures/stonefloor02_normal_astc_8x8_unorm.ktx
    textures/stonefloor02_normal_bc3_unorm.ktx
    textures/stonefloor02_normal_etc2_unorm.ktx
)
