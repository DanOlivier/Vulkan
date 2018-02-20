buildExample(shadowmappingomni shadowmappingomni.cpp)

compile_shaders(shadowmappingomni
    shaders/scene.vert
    shaders/scene.frag
    shaders/cubemapdisplay.vert
    shaders/cubemapdisplay.frag
    shaders/offscreen.vert
    shaders/offscreen.frag
)

link_assets(shadowmappingomni
    models/cube.obj
    models/shadowscene_fire.dae
)

