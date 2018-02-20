buildExample(pbribl pbribl.cpp)

compile_shaders(pbribl
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/filtercube.vert
    shaders/genbrdflut.frag
    shaders/genbrdflut.vert
    shaders/irradiancecube.frag
    shaders/pbribl.frag
    shaders/pbribl.vert
    shaders/prefilterenvmap.frag
    shaders/skybox.frag
    shaders/skybox.vert
)

copy_assets(pbribl
    models/cube.obj
    models/geosphere.obj
    models/teapot.dae
    models/torusknot.obj
    models/venus.fbx
    
    textures/hdr/pisa_cube.ktx
)
