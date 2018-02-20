buildExample(computeheadless computeheadless.cpp)

compile_shaders(computeheadless
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/headless.comp
)
