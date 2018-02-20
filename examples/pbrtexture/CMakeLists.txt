buildExample(pbrtexture main.cpp)

compile_shaders(pbrtexture
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/filtercube.vert
    shaders/genbrdflut.frag
    shaders/genbrdflut.vert
    shaders/irradiancecube.frag
    shaders/pbrtexture.frag
    shaders/pbrtexture.vert
    shaders/prefilterenvmap.frag
    shaders/skybox.frag
    shaders/skybox.vert
)

copy_assets(pbrtexture
    textures/hdr/gcanyon_cube.ktx
    models/cube.obj
    models/cerberus/cerberus.fbx
    models/cerberus/albedo.ktx
    models/cerberus/normal.ktx
    models/cerberus/ao.ktx
    models/cerberus/metallic.ktx
    models/cerberus/roughness.ktx
)
