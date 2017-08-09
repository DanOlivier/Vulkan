buildExample(computecloth computecloth.cpp)

compile_shaders(computecloth
    shaders/cloth.comp
    shaders/cloth.frag
    shaders/cloth.vert
    shaders/sphere.frag
    shaders/sphere.vert
)
