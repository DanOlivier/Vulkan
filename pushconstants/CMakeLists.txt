buildExample(pushconstants pushconstants.cpp)

compile_shaders(pushconstants
    shaders/lights.frag
    shaders/lights.vert
)

copy_assets(pushconstants
    models/samplescene.dae
)

