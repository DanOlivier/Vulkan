buildExample(debugmarker debugmarker.cpp)

compile_shaders(debugmarker
    shaders/colorpass.frag
    shaders/colorpass.vert
    shaders/postprocess.frag
    shaders/postprocess.vert
    shaders/toon.frag
    shaders/toon.vert
)

copy_assets(debugmarker
    models/treasure_smooth.dae
    models/treasure_glow.dae
)

