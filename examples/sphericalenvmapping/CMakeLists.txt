buildExample(sphericalenvmapping sphericalenvmapping.cpp)

compile_shaders(sphericalenvmapping
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/sem.frag
    shaders/sem.vert
)

link_assets(sphericalenvmapping
    models/chinesedragon.dae
    textures/matcap_array_rgba.ktx
)

