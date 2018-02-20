buildExample(sphericalenvmapping sphericalenvmapping.cpp)

compile_shaders(sphericalenvmapping
    shaders/sem.frag
    shaders/sem.vert
)

link_assets(sphericalenvmapping
    models/chinesedragon.dae
    textures/matcap_array_rgba.ktx
)

