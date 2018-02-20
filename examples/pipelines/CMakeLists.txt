buildExample(pipelines pipelines.cpp)

compile_shaders(pipelines
    shaders/phong.frag
    shaders/phong.vert
    shaders/toon.frag
    shaders/toon.vert
    shaders/wireframe.frag
    shaders/wireframe.vert
)

link_assets(pipelines
    models/treasure_smooth.dae
)

