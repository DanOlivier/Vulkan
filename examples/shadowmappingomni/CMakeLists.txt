buildExample(shadowmappingomni shadowmappingomni.cpp)

compile_shaders(shadowmappingomni
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

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

