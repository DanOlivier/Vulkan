buildExample(shadowmappingomni shadowmappingomni.cpp)

compile_shaders(shadowmappingomni
    shaders/scene.vert
    shaders/scene.frag
)

copy_assets(shadowmappingomni
    models/cube.obj
    models/shadowscene_fire.dae
)

