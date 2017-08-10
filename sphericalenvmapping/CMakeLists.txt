buildExample(sphericalenvmapping sphericalenvmapping.cpp)

compile_shaders(sphericalenvmapping
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/sem.frag
    shaders/sem.vert
)

copy_assets(sphericalenvmapping
    models/chinesedragon.dae
    textures/matcap_array_rgba.ktx
)

