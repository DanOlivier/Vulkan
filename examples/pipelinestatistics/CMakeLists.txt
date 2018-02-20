buildExample(pipelinestatistics pipelinestatistics.cpp)

compile_shaders(pipelinestatistics
    shaders/scene.frag
    shaders/scene.vert
)

link_assets(pipelinestatistics
    models/geosphere.obj
    models/teapot.dae
    models/torusknot.obj
    models/venus.fbx
)
