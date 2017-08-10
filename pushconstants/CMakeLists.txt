buildExample(pushconstants pushconstants.cpp)

compile_shaders(pushconstants
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/lights.frag
    shaders/lights.vert
)

copy_assets(pushconstants
    models/samplescene.dae
)

