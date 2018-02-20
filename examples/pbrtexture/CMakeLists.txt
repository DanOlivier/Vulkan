buildExample(pbrtexture main.cpp)

compile_shaders(pbrtexture
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

link_assets(pbrtexture
    textures/hdr/gcanyon_cube.ktx
    models/cube.obj
    models/cerberus/cerberus.fbx
    models/cerberus/albedo.ktx
    models/cerberus/normal.ktx
    models/cerberus/ao.ktx
    models/cerberus/metallic.ktx
    models/cerberus/roughness.ktx
)
