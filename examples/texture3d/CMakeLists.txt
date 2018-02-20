buildExample(texture3d texture3d.cpp)

compile_shaders(texture3d
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/texture3d.frag
    shaders/texture3d.vert
)
