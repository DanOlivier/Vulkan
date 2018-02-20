buildExample(pipelinestatistics pipelinestatistics.cpp)

compile_shaders(pipelinestatistics
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/scene.frag
    shaders/scene.vert
)

copy_assets(pipelinestatistics
    models/geosphere.obj
    models/teapot.dae
    models/torusknot.obj
    models/venus.fbx
)