buildExample(viewportarray viewportarray.cpp)

compile_shaders(viewportarray
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/multiview.geom
    shaders/scene.frag
    shaders/scene.vert
)

copy_assets(viewportarray
    models/sampleroom.dae
)
