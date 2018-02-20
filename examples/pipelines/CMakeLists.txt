buildExample(pipelines pipelines.cpp)

compile_shaders(pipelines
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/phong.frag
    shaders/phong.vert
    shaders/toon.frag
    shaders/toon.vert
    shaders/wireframe.frag
    shaders/wireframe.vert
)

copy_assets(pipelines
    models/treasure_smooth.dae
)

