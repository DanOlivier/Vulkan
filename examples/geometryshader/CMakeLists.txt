buildExample(geometryshader geometryshader.cpp)

compile_shaders(geometryshader
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/base.frag
    shaders/base.vert
    shaders/mesh.frag
    shaders/mesh.vert
    shaders/normaldebug.geom
)

copy_assets(geometryshader
    models/suzanne.obj
)

