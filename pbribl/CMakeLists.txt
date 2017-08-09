buildExample(pbribl pbribl.cpp)

compile_shaders(pbribl
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
