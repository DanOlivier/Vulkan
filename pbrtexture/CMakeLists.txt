buildExample(pbrtexture main.cpp)

compile_shaders(pbrtexture
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

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
)
