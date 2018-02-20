buildExample(pushconstants pushconstants.cpp)

compile_shaders(pushconstants
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/lights.frag
    shaders/lights.vert
)

copy_assets(pushconstants
    models/samplescene.dae
)

