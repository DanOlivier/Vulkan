buildExample(computecloth computecloth.cpp)

compile_shaders(computecloth
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/cloth.comp
    shaders/cloth.frag
    shaders/cloth.vert
    shaders/sphere.frag
    shaders/sphere.vert
)

copy_assets(computecloth
    models/geosphere.obj
    textures/vulkan_cloth_rgba.ktx
)
