buildExample(hdr hdr.cpp)

compile_shaders(hdr
    shaders/bloom.frag
    shaders/bloom.vert
    shaders/composition.frag
    shaders/composition.vert
    shaders/gbuffer.frag
    shaders/gbuffer.vert
)

copy_assets(hdr
    models/cube.obj
    models/geosphere.obj
    models/teapot.dae
    models/torusknot.obj
    models/venus.fbx
    textures/hdr/uffizi_cube.ktx
)

