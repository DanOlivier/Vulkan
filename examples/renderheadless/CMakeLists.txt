buildExample(renderheadless renderheadless.cpp)

compile_shaders(renderheadless
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/triangle.frag
    shaders/triangle.vert
)
