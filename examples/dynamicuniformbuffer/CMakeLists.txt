buildExample(dynamicuniformbuffer dynamicuniformbuffer.cpp)

compile_shaders(dynamicuniformbuffer
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/base.frag
    shaders/base.vert
)
