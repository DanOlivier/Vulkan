buildExample(stencilbuffer stencilbuffer.cpp)

compile_shaders(stencilbuffer
    shaders/toon.frag
    shaders/toon.vert
    shaders/outline.frag
    shaders/outline.vert
)

link_assets(stencilbuffer
    models/venus.fbx
)

