buildExample(gears gears.cpp vulkangear.cpp)

compile_shaders(gears
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/gears.frag
    shaders/gears.vert
)
