buildExample(timestampquery timestampquery.cpp)

compile_shaders(timestampquery
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/mesh.vert
    shaders/mesh.frag

    shaders/simple.vert
    shaders/simple.frag

    shaders/occluder.vert
    shaders/occluder.frag
)

copy_assets(timestampquery
    models/geosphere.obj
    models/teapot.dae
    models/torusknot.obj
    models/venus.fbx
)
