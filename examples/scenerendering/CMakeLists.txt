buildExample(scenerendering scenerendering.cpp)

compile_shaders(scenerendering
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/scene.frag
    shaders/scene.vert
)

copy_assets(scenerendering
    models/sibenik/dummy_rgba_unorm.ktx
    models/sibenik/kamen_astc_8x8_unorm.ktx
    models/sibenik/kamen_bc3_unorm.ktx
    models/sibenik/kamen_etc2_unorm.ktx
    models/sibenik/KAMEN-stup_astc_8x8_unorm.ktx
    models/sibenik/KAMEN-stup_bc3_unorm.ktx
    models/sibenik/KAMEN-stup_etc2_unorm.ktx
    models/sibenik/mramor6x6_astc_8x8_unorm.ktx
    models/sibenik/mramor6x6_bc3_unorm.ktx
    models/sibenik/mramor6x6_etc2_unorm.ktx
    models/sibenik/sibenik.dae
)

