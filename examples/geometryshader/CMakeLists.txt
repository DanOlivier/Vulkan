buildExample(geometryshader geometryshader.cpp)

compile_shaders(geometryshader
    shaders/base.frag
    shaders/base.vert
    shaders/mesh.frag
    shaders/mesh.vert
    shaders/normaldebug.geom
)

link_assets(geometryshader
    models/suzanne.obj
)

