buildExample(timestampquery timestampquery.cpp)

compile_shaders(timestampquery
    shaders/mesh.vert
    shaders/mesh.frag
    shaders/simple.vert
    shaders/simple.frag
    shaders/occluder.vert
    shaders/occluder.frag
)

link_assets(timestampquery
    models/geosphere.obj
    models/teapot.dae
    models/torusknot.obj
    models/venus.fbx
)
