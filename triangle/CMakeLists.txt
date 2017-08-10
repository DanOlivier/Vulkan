buildExample(triangle triangle.cpp)

compile_shaders(triangle
    ../base/shaders/textoverlay.vert
    ../base/shaders/textoverlay.frag

    shaders/triangle.frag
    shaders/triangle.vert
)
