buildExample(deferred deferred.cpp)

compile_shaders(deferred
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/debug.frag
    shaders/debug.vert
    shaders/deferred.frag
    shaders/deferred.vert
    shaders/mrt.frag
    shaders/mrt.vert
)

copy_assets(deferred
    models/plane.obj

    models/armor/armor.dae
    models/armor/armor.dae
    models/armor/color_astc_8x8_unorm.ktx
    models/armor/color_bc3_unorm.ktx
    models/armor/color_etc2_unorm.ktx
    models/armor/normal_astc_8x8_unorm.ktx
    models/armor/normal_bc3_unorm.ktx
    models/armor/normal_etc2_unorm.ktx

    textures/stonefloor01_color_astc_8x8_unorm.ktx
    textures/stonefloor01_color_bc3_unorm.ktx
    textures/stonefloor01_color_etc2_unorm.ktx
    textures/stonefloor01_normal_astc_8x8_unorm.ktx
    textures/stonefloor01_normal_bc3_unorm.ktx
    textures/stonefloor01_normal_etc2_unorm.ktx
)

