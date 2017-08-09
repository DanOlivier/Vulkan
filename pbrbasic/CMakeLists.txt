buildExample(pbrbasic pbrbasic.cpp)

compile_shaders(pbrbasic
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
