buildExample(occlusionquery occlusionquery.cpp)

compile_shaders(occlusionquery
    shaders/mesh.frag
    shaders/mesh.vert
    shaders/occluder.frag
    shaders/occluder.vert
    shaders/simple.frag
    shaders/simple.vert
)

copy_assets(occlusionquery
    models/plane_z.3ds
    models/teapot.3ds
    models/sphere.3ds
)

