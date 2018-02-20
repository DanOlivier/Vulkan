buildExample(pushconstants pushconstants.cpp)

compile_shaders(pushconstants
    shaders/lights.frag
    shaders/lights.vert
)

link_assets(pushconstants
    models/samplescene.dae
)

