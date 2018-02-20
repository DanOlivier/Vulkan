buildExample(pbrbasic pbrbasic.cpp)

compile_shaders(pbrbasic
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/pbr.frag
    shaders/pbr.vert
)

copy_assets(pbrbasic
    models/cube.obj
    models/geosphere.obj
    models/teapot.dae
    models/torusknot.obj
    models/venus.fbx
)
