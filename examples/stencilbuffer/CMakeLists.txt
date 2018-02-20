buildExample(stencilbuffer stencilbuffer.cpp)

compile_shaders(stencilbuffer
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/toon.frag
    shaders/toon.vert

    shaders/outline.frag
    shaders/outline.vert
)

link_assets(stencilbuffer
    models/venus.fbx
)

