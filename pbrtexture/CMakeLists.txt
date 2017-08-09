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
